import { describe, expect, test } from "bun:test";
import { captureChatDefaults, MAX_CHAT_ROUNDS, normalizeChatJobs, runChatTurn, type ChatMessage } from "./chatTurn";
import { DEFAULT_SETTINGS, MAX_JOBS, type RenderRequest } from "./domain";

const defaults = captureChatDefaults({ ...DEFAULT_SETTINGS, seed: "100" });
const loras = [{ name: "Ink", path: "/models/ink.safetensors", size_bytes: 20 }];
const history: ChatMessage[] = [{ role: "user", content: "Make the images" }];
const call = (id: string, jobs: unknown, name = "queue_txt2img") => ({ id, type: "function", function: { name, arguments: JSON.stringify({ jobs }) } });
const response = (calls: unknown[] = [], finish = calls.length ? "tool_calls" : "stop") => ({ choices: [{ finish_reason: finish, message: { role: "assistant", content: calls.length ? null : "Prepared the complete batch.", ...(calls.length ? { tool_calls: calls } : {}) } }] });

function harness(responses: Record<string, unknown>[]) {
  const requests: unknown[][] = [];
  const committed: RenderRequest[][] = [];
  const events: string[] = [];
  return { requests, committed, events, run: () => runChatTurn({ history, defaults, loras,
    chat: async (messages) => { requests.push(structuredClone(messages)); events.push("chat"); const next = responses.shift(); if (!next) throw new Error("No fixture response"); return next; },
    onQueue: async (jobs) => { events.push("commit"); committed.push(jobs); },
    onStaged: (count) => events.push(`staged:${count}`),
  }) };
}

describe("chat job validation", () => {
  test("distinct jobs and large n expand to independent single-image requests", () => {
    const result = normalizeChatJobs({ jobs: [{ prompt: "first", n: 20, seed: 7 }, { prompt: "second", loras: [{ path: loras[0]!.path, weight: -4 }] }] }, defaults, loras);
    expect(result).toHaveLength(21);
    expect(result.slice(0, 20).map((request) => request.seed)).toEqual(Array.from({ length: 20 }, (_, i) => i + 7));
    expect(result.every((request) => request.n === 1)).toBe(true);
    expect(result[20]).toEqual({ prompt: "second", width: 1024, height: 1024, steps: 8, seed: 100, n: 1, loras: [{ name: loras[0]!.path, weight: -4 }] });
    expect(result[0]!.loras).not.toBe(result[1]!.loras);
  });
  test("blank-seed defaults advance across jobs and rounds, pinned seeds stay matched", () => {
    const randomDefaults = { ...defaults, advanceSeed: true };
    expect(normalizeChatJobs({ jobs: [{ prompt: "first", n: 2 }, { prompt: "second" }] }, randomDefaults, [], MAX_JOBS, 5).map((job) => job.seed)).toEqual([105, 106, 107]);
    expect(normalizeChatJobs({ jobs: [{ prompt: "first" }, { prompt: "second" }] }, defaults, []).map((job) => job.seed)).toEqual([100, 100]);
  });
  test("defaults are captured from settings, including active LoRAs", () => {
    const settings = { ...DEFAULT_SETTINGS, seed: "42", loras: [{ name: loras[0]!.path, weight: 0.5 }] };
    const captured = captureChatDefaults(settings);
    settings.width = 512; settings.loras[0]!.weight = 1;
    const [job] = normalizeChatJobs({ jobs: [{ prompt: "first" }] }, captured, loras);
    expect(job!.width).toBe(1024); expect(job!.seed).toBe(42); expect(job!.loras[0]!.weight).toBe(0.5);
  });
  test.each([
    null, [], {}, { prompt: "old schema" }, { jobs: [] }, { jobs: [{ prompt: 3 }] },
    { jobs: [{ prompt: "a", n: "2" }] }, { jobs: [{ prompt: "a", width: "1024" }] },
    { jobs: [{ prompt: "a", steps: null }] }, { jobs: [{ prompt: "a", strength: 0.5 }] },
    { jobs: [{ prompt: "a" }], extra: true }, { jobs: [{ prompt: "a", n: 0 }] },
    { jobs: [{ prompt: "a", n: 1.2 }] }, { jobs: [{ prompt: "a", n: MAX_JOBS + 1 }] },
    { jobs: [{ prompt: "a", seed: Number.MAX_SAFE_INTEGER, n: 2 }] },
    { jobs: [{ prompt: "a", loras: null }] }, { jobs: [{ prompt: "a", loras: [{ path: "/unlisted", weight: 1 }] }] },
    { jobs: [{ prompt: "a", loras: [{ path: loras[0]!.path, weight: "1" }] }] },
    { jobs: [{ prompt: "a", loras: [{ path: loras[0]!.path, weight: 4.1 }] }] },
    { jobs: [{ prompt: "a", loras: [{ path: loras[0]!.path, weight: Infinity }] }] },
    { jobs: [{ prompt: "a", loras: [{ path: loras[0]!.path, weight: 1, extra: 1 }] }] },
  ].map((value) => [value]))("rejects malformed or unsafe jobs: %j", (value) => {
    expect(() => normalizeChatJobs(value, defaults, loras)).toThrow();
  });
  test("allows consecutive seeds ending exactly at the safe integer maximum", () => {
    const result = normalizeChatJobs({ jobs: [{ prompt: "a", seed: Number.MAX_SAFE_INTEGER - 1, n: 2 }] }, defaults, []);
    expect(result.map((job) => job.seed)).toEqual([Number.MAX_SAFE_INTEGER - 1, Number.MAX_SAFE_INTEGER]);
  });
  test("validates inherited LoRAs against the current server list", () => {
    expect(() => normalizeChatJobs({ jobs: [{ prompt: "a" }] }, { ...defaults, loras: [{ name: "/removed", weight: 1 }] }, loras)).toThrow("exact path");
  });
});

describe("complete assistant turns", () => {
  test("honors every tool call and more than four rounds, commits once after final reply", async () => {
    const h = harness([
      response([call("one", [{ prompt: "a", n: 2 }]), call("two", [{ prompt: "b" }])]),
      ...Array.from({ length: 5 }, (_, i) => response([call(`more-${i}`, [{ prompt: `extra-${i}` }])])),
      response(),
    ]);
    const result = await h.run();
    expect(h.requests).toHaveLength(7);
    expect(h.committed).toHaveLength(1); expect(h.committed[0]).toHaveLength(8);
    expect(h.events.at(-1)).toBe("commit"); expect(result.queued).toBe(8);
    expect(result.history.map((message) => message.role)).toEqual(["user", "assistant", "tool", "tool", ...Array(5).fill(["assistant", "tool"]).flat(), "assistant"]);
    for (const message of result.history.filter((message) => message.role === "tool")) {
      expect(JSON.parse(message.content!)).toMatchObject({ staged: true });
      expect(JSON.parse(message.content!)).not.toHaveProperty("queued");
    }
    expect(h.requests[1]!.map((message) => (message as ChatMessage).role)).toEqual(["system", "user", "assistant", "tool", "tool"]);
    expect(history).toHaveLength(1);
  });
  test("ordinary final text needs no queue commit", async () => {
    const h = harness([response()]); expect((await h.run()).queued).toBe(0); expect(h.committed).toEqual([]);
  });
  test("tool calls on stop are processed, then require a separate final stop", async () => {
    const h = harness([response([call("one", [{ prompt: "a" }])], "stop"), response()]);
    await h.run(); expect(h.committed).toHaveLength(1); expect(h.requests).toHaveLength(2);
  });
  test.each([
    response([], "length"), response([], "content_filter"), response([], "tool_calls"),
    { choices: [{ message: { role: "assistant", content: "missing finish" } }] },
    { choices: [{ finish_reason: "stop", message: { role: "user", content: "wrong role" } }] },
    { choices: [{ finish_reason: "stop", message: { role: "assistant", content: "broken", tool_calls: {} } }] },
    { choices: [{ finish_reason: "stop", message: { role: "assistant", content: null } }] },
    response([call("bad", [{ prompt: "a" }], "other_tool")]),
    response([call("bad", [{ prompt: "a", seed: -1 }])]),
    response([{ id: "bad", type: "function", function: { name: "queue_txt2img", arguments: "{truncated" } }]),
  ])("discards staged jobs on invalid later response: %j", async (bad) => {
    const h = harness([response([call("one", [{ prompt: "a" }])]), bad]);
    await expect(h.run()).rejects.toThrow(); expect(h.committed).toEqual([]); expect(history).toHaveLength(1);
  });
  test("a malformed later tool in the same response commits nothing", async () => {
    const h = harness([response([call("one", [{ prompt: "a" }]), call("two", [{ prompt: "b", n: -1 }])])]);
    await expect(h.run()).rejects.toThrow(); expect(h.committed).toEqual([]);
  });
  test("caps total expanded images across calls", async () => {
    const h = harness([response([call("one", [{ prompt: "a", n: MAX_JOBS }]), call("two", [{ prompt: "b" }])])]);
    await expect(h.run()).rejects.toThrow("at most 1000"); expect(h.committed).toEqual([]);
  });
  test("refuses reused call IDs", async () => {
    const h = harness([response([call("one", [{ prompt: "a" }])]), response([call("one", [{ prompt: "b" }])])]);
    await expect(h.run()).rejects.toThrow("reused"); expect(h.committed).toEqual([]);
  });
  test("round cap is a failure and never submits the partial batch", async () => {
    const h = harness(Array.from({ length: MAX_CHAT_ROUNDS }, (_, i) => response([call(`round-${i}`, [{ prompt: "a" }])])));
    await expect(h.run()).rejects.toThrow("16 rounds"); expect(h.committed).toEqual([]); expect(h.requests).toHaveLength(16);
  });
  test("transport failure after staging commits nothing", async () => {
    const h = harness([response([call("one", [{ prompt: "a" }])])]);
    await expect(h.run()).rejects.toThrow("No fixture"); expect(h.committed).toEqual([]);
  });
  test("queue persistence failure does not retry or publish transcript", async () => {
    let commits = 0; let rounds = 0;
    await expect(runChatTurn({ history, defaults, loras, chat: async () => rounds++ ? response() : response([call("one", [{ prompt: "a" }])]), onQueue: async () => { commits++; throw new Error("Persistence failed"); } })).rejects.toThrow("Persistence failed");
    expect(commits).toBe(1); expect(history).toHaveLength(1);
  });
});
