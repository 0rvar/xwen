import { describe, expect, test } from "bun:test";
import { createBatchDraft } from "./batchDraft";
import type { PlannedJob } from "./domain";

describe("batch drafts", () => {
  test("deduplicates image data while leaving planned jobs unchanged", () => {
    const source = "data:image/png;base64,c291cmNl";
    const mask = "data:image/png;base64,bWFzaw==";
    const jobs: PlannedJob[] = [0, 1].map((index) => ({
      id: `display-${index}`,
      request: {
        prompt: `job ${index}`,
        width: 512,
        height: 512,
        steps: 8,
        seed: index,
        n: 1,
        loras: [],
        init_image: source,
        mask,
        control: { image: source, preprocess: "none", scale: 0.5, start: 0, end: 1 },
      },
      context: { mode: "inpaint", batch_index: index, axes: { seed: index }, repeat_index: 0 },
    }));

    const draft = createBatchDraft(jobs, { mode: "matrix", axes: [{ parameter: "seed", values: "0,1" }], count: 1 });

    expect(draft.inputs).toEqual({ input_0: source, input_1: mask });
    expect(draft.jobs[0]!.request).toMatchObject({ init_image: "input_0", mask: "input_1", control: { image: "input_0" } });
    expect(draft.jobs[1]!.request).toMatchObject({ init_image: "input_0", mask: "input_1", control: { image: "input_0" } });
    expect(jobs[0]!.request).toMatchObject({ init_image: source, mask, control: { image: source } });
  });
});
