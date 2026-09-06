// llama-logits-all — dump llama.cpp's logits at EVERY position of a prompt.
//
// The parity oracle for a full-precision, per-position comparison against xwen's
// own forward pass: xwen dumps [n_tokens, n_vocab] f32 for the same ids and the
// two files are compared row by row. llama-cli and llama-eval-callback cannot do
// this — the first only ever exposes the last position's logits, the second dumps
// intermediate tensors, not the head.
//
// Every token in the batch carries logits = 1, so one decode of a ubatch yields
// one logits row per position in it. The rows are written out as they are
// produced, so a 4k-token prompt never needs its full 2.4 GB in memory.
//
// Outputs, given --out <prefix>:
//   <prefix>.f32         raw little-endian f32 [n_tokens, n_vocab], row-major
//   <prefix>.json        run metadata: shapes, backend placement, KV types, batch
//                        geometry, flash-attn setting, threads, model identity
//   <prefix>.argmax.json argmax token id per position + top-5 at the last position,
//                        so a cross-check needs no f32 parsing
//   <prefix>.argmax.txt  the argmax stream detokenized, raw bytes, for eyeballing
//
// Build: see scripts/build-llamacpp.sh, which owns the exact compile line.

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

namespace {

[[noreturn]] void die(const std::string & msg) {
    fprintf(stderr, "llama-logits-all: error: %s\n", msg.c_str());
    exit(1);
}

// llama.cpp's log callback delivers fragments, not whole lines. Reassemble them so
// the "load_tensors:" lines — the ground truth for which device holds the weights —
// can go into the JSON sidecar verbatim.
struct LogCapture {
    std::string pending;
    std::vector<std::string> load_tensors_lines;
};

LogCapture g_log;

void log_cb(enum ggml_log_level level, const char * text, void * /*user_data*/) {
    if (level != GGML_LOG_LEVEL_NONE) {
        fputs(text, stderr);
    }
    g_log.pending += text;
    size_t nl;
    while ((nl = g_log.pending.find('\n')) != std::string::npos) {
        std::string line = g_log.pending.substr(0, nl);
        g_log.pending.erase(0, nl + 1);
        if (line.rfind("load_tensors:", 0) == 0) {
            g_log.load_tensors_lines.push_back(line);
        }
    }
}

std::string json_escape(const std::string & s) {
    std::string out;
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20 || c >= 0x80) {
                    // Escape as a latin-1 codepoint: keeps the JSON valid even when the
                    // text is a fragment of a multi-byte sequence. Readable text lives
                    // in the .argmax.txt sidecar, which needs no escaping at all.
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", (unsigned) c);
                    out += buf;
                } else {
                    out += (char) c;
                }
        }
    }
    return out;
}

std::string arg_after(int argc, char ** argv, int & i, const char * name) {
    if (i + 1 >= argc) {
        die(std::string("missing value for ") + name);
    }
    return argv[++i];
}

const char * kUsage =
    "usage: llama-logits-all --model <gguf> --ids <file> --out <prefix>\n"
    "                       [--n-gpu-layers N] [--batch N] [--kv-type f16|f32]\n"
    "                       [--flash-attn on|off|auto] [--threads N]\n"
    "                       [--expect-vocab N] [--model-sha256 HEX] [--llamacpp-commit SHA]\n"
    "\n"
    "  --ids           file of whitespace-separated token ids (no BOS is added)\n"
    "  --out           output prefix; writes .f32, .json, .argmax.json, .argmax.txt\n"
    "  --n-gpu-layers  layers offloaded to the GPU (default 0 = CPU only)\n"
    "  --batch         cap on n_batch/n_ubatch (default 512)\n"
    "  --kv-type       KV cache type for both K and V (default f16)\n"
    "  --flash-attn    flash attention: on, off or auto (default auto)\n"
    "  --threads       CPU threads (default 8)\n"
    "  --expect-vocab  fail unless the model's vocabulary has exactly this many tokens\n"
    "  --model-sha256  recorded verbatim in the JSON sidecar; not verified here\n"
    "  --llamacpp-commit  recorded verbatim in the JSON sidecar\n";

} // namespace

int main(int argc, char ** argv) {
    std::string model_path;
    std::string ids_path;
    std::string out_prefix;
    std::string kv_type_name = "f16";
    std::string flash_attn_name = "auto";
    std::string model_sha256;
    std::string llamacpp_commit;
    int n_gpu_layers = 0;
    int batch_cap    = 512;
    int n_threads    = 8;
    int expect_vocab = 0;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--help" || a == "-h") {
            fputs(kUsage, stdout);
            return 0;
        } else if (a == "--model") {
            model_path = arg_after(argc, argv, i, "--model");
        } else if (a == "--ids") {
            ids_path = arg_after(argc, argv, i, "--ids");
        } else if (a == "--out") {
            out_prefix = arg_after(argc, argv, i, "--out");
        } else if (a == "--n-gpu-layers" || a == "-ngl") {
            n_gpu_layers = atoi(arg_after(argc, argv, i, "--n-gpu-layers").c_str());
        } else if (a == "--batch") {
            batch_cap = atoi(arg_after(argc, argv, i, "--batch").c_str());
        } else if (a == "--kv-type") {
            kv_type_name = arg_after(argc, argv, i, "--kv-type");
        } else if (a == "--flash-attn") {
            flash_attn_name = arg_after(argc, argv, i, "--flash-attn");
        } else if (a == "--threads") {
            n_threads = atoi(arg_after(argc, argv, i, "--threads").c_str());
        } else if (a == "--expect-vocab") {
            expect_vocab = atoi(arg_after(argc, argv, i, "--expect-vocab").c_str());
        } else if (a == "--model-sha256") {
            model_sha256 = arg_after(argc, argv, i, "--model-sha256");
        } else if (a == "--llamacpp-commit") {
            llamacpp_commit = arg_after(argc, argv, i, "--llamacpp-commit");
        } else {
            fputs(kUsage, stderr);
            die("unknown argument: " + a);
        }
    }

    if (model_path.empty() || ids_path.empty() || out_prefix.empty()) {
        fputs(kUsage, stderr);
        die("--model, --ids and --out are all required");
    }
    if (batch_cap <= 0)  die("--batch must be positive");
    if (n_threads <= 0)  die("--threads must be positive");

    ggml_type kv_type;
    if      (kv_type_name == "f16") kv_type = GGML_TYPE_F16;
    else if (kv_type_name == "f32") kv_type = GGML_TYPE_F32;
    else die("--kv-type must be f16 or f32, got " + kv_type_name);

    llama_flash_attn_type flash_attn;
    if      (flash_attn_name == "auto") flash_attn = LLAMA_FLASH_ATTN_TYPE_AUTO;
    else if (flash_attn_name == "on")   flash_attn = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    else if (flash_attn_name == "off")  flash_attn = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    else die("--flash-attn must be on, off or auto, got " + flash_attn_name);

    // Read the ids.
    std::vector<llama_token> ids;
    {
        FILE * f = fopen(ids_path.c_str(), "rb");
        if (!f) die("cannot open ids file: " + ids_path);
        long long v;
        while (fscanf(f, "%lld", &v) == 1) {
            ids.push_back((llama_token) v);
        }
        // Anything left unconsumed means a non-numeric token in the file.
        int c;
        while ((c = fgetc(f)) != EOF) {
            if (!isspace(c)) {
                fclose(f);
                die("ids file contains non-numeric data: " + ids_path);
            }
        }
        fclose(f);
    }
    if (ids.empty()) die("ids file is empty: " + ids_path);
    const int n_tokens = (int) ids.size();

    llama_log_set(log_cb, nullptr);
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) die("llama_model_load_from_file failed for " + model_path);

    const llama_vocab * vocab = llama_model_get_vocab(model);
    if (!vocab) die("llama_model_get_vocab returned null");
    const int n_vocab = llama_vocab_n_tokens(vocab);
    if (n_vocab <= 0) die("model reports a non-positive vocabulary size");
    if (expect_vocab > 0 && n_vocab != expect_vocab) {
        die("vocabulary size mismatch: model has " + std::to_string(n_vocab) +
            " tokens, --expect-vocab said " + std::to_string(expect_vocab));
    }
    for (int i = 0; i < n_tokens; i++) {
        if (ids[i] < 0 || ids[i] >= n_vocab) {
            die("token id out of range at position " + std::to_string(i) + ": " +
                std::to_string(ids[i]) + " (vocab " + std::to_string(n_vocab) + ")");
        }
    }

    const int n_ubatch = std::min(n_tokens, batch_cap);
    // Context sized to hold the whole prompt, rounded up to a multiple of 256.
    const int n_ctx = ((n_tokens + 255) / 256) * 256;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = n_ctx;
    cparams.n_batch         = n_ubatch;
    cparams.n_ubatch        = n_ubatch;
    cparams.n_seq_max       = 1;
    // One logits row per token in the ubatch: the whole point of this tool.
    cparams.n_outputs_max   = n_ubatch;
    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads;
    cparams.type_k          = kv_type;
    cparams.type_v          = kv_type;
    cparams.flash_attn_type = flash_attn;
    cparams.embeddings      = false;
    cparams.no_perf         = false;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) die("llama_init_from_model failed");

    const std::string f32_path = out_prefix + ".f32";
    FILE * f32_out = fopen(f32_path.c_str(), "wb");
    if (!f32_out) die("cannot open output file: " + f32_path);

    std::vector<llama_token> argmax(n_tokens);
    std::vector<int>   top5_ids(5, -1);
    std::vector<float> top5_logits(5, 0.0f);

    llama_batch batch = llama_batch_init(n_ubatch, 0, 1);

    const int64_t t_start_us = llama_time_us();

    for (int start = 0; start < n_tokens; start += n_ubatch) {
        const int m = std::min(n_ubatch, n_tokens - start);
        batch.n_tokens = m;
        for (int i = 0; i < m; i++) {
            batch.token[i]     = ids[start + i];
            batch.pos[i]       = start + i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = 1; // logits for EVERY position
        }

        const int rc = llama_decode(ctx, batch);
        if (rc != 0) {
            die("llama_decode returned " + std::to_string(rc) + " at position " +
                std::to_string(start));
        }

        for (int i = 0; i < m; i++) {
            const float * row = llama_get_logits_ith(ctx, i);
            if (!row) {
                die("llama_get_logits_ith returned null for batch index " + std::to_string(i) +
                    " at position " + std::to_string(start + i));
            }
            if (fwrite(row, sizeof(float), (size_t) n_vocab, f32_out) != (size_t) n_vocab) {
                die("short write to " + f32_path);
            }

            int best = 0;
            for (int t = 1; t < n_vocab; t++) {
                if (row[t] > row[best]) best = t;
            }
            argmax[start + i] = best;

            if (start + i == n_tokens - 1) {
                std::vector<int> order(n_vocab);
                std::iota(order.begin(), order.end(), 0);
                std::partial_sort(order.begin(), order.begin() + 5, order.end(),
                                  [&](int a, int b) { return row[a] > row[b]; });
                for (int k = 0; k < 5; k++) {
                    top5_ids[k]    = order[k];
                    top5_logits[k] = row[order[k]];
                }
            }
        }
    }

    const double elapsed_s = (llama_time_us() - t_start_us) / 1e6;

    if (fclose(f32_out) != 0) die("failed to close " + f32_path);

    llama_batch_free(batch);

    // Detokenize the argmax stream and the last position's top-5 into a raw text
    // sidecar: no JSON escaping, so multi-byte pieces survive intact.
    auto piece = [&](llama_token tok) {
        char buf[256];
        const int n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, true);
        if (n < 0) return std::string("<piece too long>");
        return std::string(buf, n);
    };

    {
        const std::string txt_path = out_prefix + ".argmax.txt";
        FILE * f = fopen(txt_path.c_str(), "wb");
        if (!f) die("cannot open output file: " + txt_path);
        fprintf(f, "# argmax continuation, one piece per input position\n");
        for (int i = 0; i < n_tokens; i++) {
            const std::string p = piece(argmax[i]);
            fwrite(p.data(), 1, p.size(), f);
        }
        fprintf(f, "\n# top-5 at the last position (id, logit, piece)\n");
        for (int k = 0; k < 5; k++) {
            fprintf(f, "%d\t%.6f\t", top5_ids[k], top5_logits[k]);
            const std::string p = piece(top5_ids[k]);
            fwrite(p.data(), 1, p.size(), f);
            fputc('\n', f);
        }
        if (fclose(f) != 0) die("failed to close " + txt_path);
    }

    {
        const std::string am_path = out_prefix + ".argmax.json";
        FILE * f = fopen(am_path.c_str(), "wb");
        if (!f) die("cannot open output file: " + am_path);
        fprintf(f, "{\n  \"n_tokens\": %d,\n  \"argmax\": [", n_tokens);
        for (int i = 0; i < n_tokens; i++) {
            fprintf(f, "%s%d", i ? ", " : "", argmax[i]);
        }
        fprintf(f, "],\n  \"top5_last\": [\n");
        for (int k = 0; k < 5; k++) {
            fprintf(f, "    {\"id\": %d, \"logit\": %.9g, \"piece\": \"%s\"}%s\n",
                    top5_ids[k], top5_logits[k], json_escape(piece(top5_ids[k])).c_str(),
                    k == 4 ? "" : ",");
        }
        fprintf(f, "  ]\n}\n");
        if (fclose(f) != 0) die("failed to close " + am_path);
    }

    {
        const std::string meta_path = out_prefix + ".json";
        FILE * f = fopen(meta_path.c_str(), "wb");
        if (!f) die("cannot open output file: " + meta_path);
        char desc[256] = {0};
        llama_model_desc(model, desc, sizeof(desc));

        fprintf(f, "{\n");
        fprintf(f, "  \"tool\": \"llama-logits-all\",\n");
        fprintf(f, "  \"n_tokens\": %d,\n", n_tokens);
        fprintf(f, "  \"n_vocab\": %d,\n", n_vocab);
        fprintf(f, "  \"dtype\": \"f32\",\n");
        fprintf(f, "  \"layout\": \"[n_tokens, n_vocab] row-major little-endian\",\n");
        fprintf(f, "  \"model_path\": \"%s\",\n", json_escape(model_path).c_str());
        fprintf(f, "  \"model_sha256\": \"%s\",\n", json_escape(model_sha256).c_str());
        fprintf(f, "  \"model_desc\": \"%s\",\n", json_escape(desc).c_str());
        fprintf(f, "  \"llamacpp_commit\": \"%s\",\n", json_escape(llamacpp_commit).c_str());
        fprintf(f, "  \"n_gpu_layers\": %d,\n", n_gpu_layers);
        fprintf(f, "  \"backend\": \"%s\",\n", n_gpu_layers == 0 ? "CPU" : "GPU-offloaded");
        fprintf(f, "  \"n_ctx\": %u,\n", llama_n_ctx(ctx));
        fprintf(f, "  \"n_batch\": %u,\n", llama_n_batch(ctx));
        fprintf(f, "  \"n_ubatch\": %u,\n", llama_n_ubatch(ctx));
        fprintf(f, "  \"type_k\": \"%s\",\n", ggml_type_name(kv_type));
        fprintf(f, "  \"type_v\": \"%s\",\n", ggml_type_name(kv_type));
        fprintf(f, "  \"flash_attn\": \"%s\",\n", flash_attn_name.c_str());
        fprintf(f, "  \"n_threads\": %d,\n", n_threads);
        fprintf(f, "  \"decode_seconds\": %.3f,\n", elapsed_s);

        fprintf(f, "  \"registered_backends\": [");
        for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
            ggml_backend_reg_t reg = ggml_backend_reg_get(i);
            fprintf(f, "%s\"%s\"", i ? ", " : "", json_escape(ggml_backend_reg_name(reg)).c_str());
        }
        fprintf(f, "],\n");

        fprintf(f, "  \"devices\": [");
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            fprintf(f, "%s{\"name\": \"%s\", \"description\": \"%s\"}",
                    i ? ", " : "",
                    json_escape(ggml_backend_dev_name(dev)).c_str(),
                    json_escape(ggml_backend_dev_description(dev)).c_str());
        }
        fprintf(f, "],\n");

        // llama.cpp's own record of where each tensor plane landed: the authority on
        // which backend actually ran the forward pass.
        fprintf(f, "  \"load_tensors_log\": [\n");
        for (size_t i = 0; i < g_log.load_tensors_lines.size(); i++) {
            fprintf(f, "    \"%s\"%s\n", json_escape(g_log.load_tensors_lines[i]).c_str(),
                    i + 1 == g_log.load_tensors_lines.size() ? "" : ",");
        }
        fprintf(f, "  ]\n}\n");
        if (fclose(f) != 0) die("failed to close " + meta_path);
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    fprintf(stderr, "llama-logits-all: wrote %d x %d f32 to %s in %.3f s\n",
            n_tokens, n_vocab, f32_path.c_str(), elapsed_s);
    return 0;
}
