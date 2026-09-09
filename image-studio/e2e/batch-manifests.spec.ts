import { expect, test } from "@playwright/test";

test("rapid submissions create separate manifests and render in submission order", async ({ page }) => {
  await page.goto("/?batchDelay=1500&renderDelay=120");
  await page.getByLabel("Prompt").fill("manifest first");
  await page.getByRole("button", { name: /^Generate/ }).click();
  await page.getByLabel("Prompt").fill("manifest second");
  await page.getByRole("button", { name: /^Generate/ }).click();

  await expect(page.getByText("2 preparing", { exact: true })).toBeVisible();
  await expect(page.getByText("2 done", { exact: true })).toBeVisible({ timeout: 7_000 });
  const preview = await page.evaluate(() => {
    const state = window as typeof window & {
      __XWEN_PREVIEW_BATCHES__?: Array<{ handle: { id: string; job_ids: string[] }; draft: { definition: { mode: string }; jobs: Array<{ request: { prompt: string } }> } }>;
      __XWEN_PREVIEW_REQUESTS__?: Array<{ prompt: string }>;
    };
    return { batches: state.__XWEN_PREVIEW_BATCHES__, prompts: state.__XWEN_PREVIEW_REQUESTS__?.map((request) => request.prompt) };
  });
  expect(preview.batches).toHaveLength(2);
  expect(preview.batches?.map((batch) => batch.handle.id)).toEqual(["preview-batch-1", "preview-batch-2"]);
  expect(preview.batches?.map((batch) => batch.draft.definition.mode)).toEqual(["single", "single"]);
  expect(preview.prompts).toEqual(["manifest first", "manifest second"]);
});

test("a batch manifest shares duplicate source and control assets", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Image", exact: true }).click();
  await page.getByRole("button", { name: "Choose image" }).click();
  await page.getByText("ControlNet", { exact: true }).click();
  const control = page.locator("details").filter({ hasText: "ControlNet" });
  await control.getByRole("button", { name: "Choose image" }).click();
  await page.getByLabel("Prompt").fill("shared source asset");
  await page.getByRole("button", { name: /^Generate/ }).click();
  await expect(page.getByText("1 done", { exact: true })).toBeVisible({ timeout: 3_000 });

  const draft = await page.evaluate(() => (window as typeof window & {
    __XWEN_PREVIEW_BATCH_DRAFTS__?: Array<{ inputs: Record<string, string>; jobs: Array<{ request: { init_image?: string; control?: { image: string } } }> }>;
  }).__XWEN_PREVIEW_BATCH_DRAFTS__?.[0]);
  expect(Object.keys(draft?.inputs ?? {})).toEqual(["input_0"]);
  expect(draft?.jobs[0]?.request.init_image).toBe("input_0");
  expect(draft?.jobs[0]?.request.control?.image).toBe("input_0");
});

test("retry uses the same persisted batch and job identifiers", async ({ page }) => {
  await page.goto("/?renderDelay=100");
  await page.getByLabel("Prompt").fill("fail preview persisted retry");
  await page.getByRole("button", { name: /^Generate/ }).click();
  await expect(page.getByText("Preview render failed as requested.")).toBeVisible();
  await page.getByRole("button", { name: "Retry" }).click();
  await expect(page.getByText("1 done", { exact: true })).toBeVisible({ timeout: 3_000 });

  const attempts = await page.evaluate(() => (window as typeof window & {
    __XWEN_PREVIEW_BATCH_RENDERS__?: Array<{ batchId: string; jobId: string }>;
  }).__XWEN_PREVIEW_BATCH_RENDERS__);
  expect(attempts).toHaveLength(2);
  expect(attempts?.[1]).toEqual(attempts?.[0]);
});

test("discard persists waiting job ids before removing them from the queue", async ({ page }) => {
  await page.goto("/?renderDelay=300");
  await page.getByLabel("Prompt").fill("discard persisted jobs");
  await page.getByLabel("Count").fill("2");
  await page.getByRole("button", { name: /^Generate/ }).click();
  await expect(page.getByText("1 running", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Stop after current" }).click();
  await page.getByLabel("Count").fill("1");
  await page.getByLabel("Prompt").fill("discard second manifest");
  await page.getByRole("button", { name: /^Generate/ }).click();
  await expect(page.getByText("2 waiting", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Discard 2 waiting" }).click();
  await expect(page.getByText("0 waiting", { exact: true })).toBeVisible();

  const discards = await page.evaluate(() => (window as typeof window & {
    __XWEN_PREVIEW_DISCARDS__?: Array<{ batchId: string; jobIds: string[] }>;
  }).__XWEN_PREVIEW_DISCARDS__);
  expect(discards).toEqual([
    { batchId: "preview-batch-1", jobIds: ["job-000002"] },
    { batchId: "preview-batch-2", jobIds: ["job-000001"] },
  ]);
});

test("a discard persistence failure keeps the waiting job in the queue", async ({ page }) => {
  await page.goto("/?renderDelay=300&failDiscard=1");
  await page.getByLabel("Prompt").fill("keep after discard failure");
  await page.getByLabel("Count").fill("2");
  await page.getByRole("button", { name: /^Generate/ }).click();
  await expect(page.getByText("1 running", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Stop after current" }).click();
  await expect(page.getByText("1 waiting", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Discard 1 waiting" }).click();
  await expect(page.getByText("Preview discard failed as requested.")).toBeVisible();
  await expect(page.getByText("1 waiting", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Discard 1 waiting" }).click();
  await expect(page.getByText("0 waiting", { exact: true })).toBeVisible();
});

test("a failed manifest does not poison later creation", async ({ page }) => {
  await page.goto("/");
  await page.getByLabel("Prompt").fill("fail manifest");
  await page.getByRole("button", { name: /^Generate/ }).click();
  await expect(page.getByText("Preview batch creation failed as requested.")).toBeVisible();
  await page.getByLabel("Prompt").fill("creation after failure");
  await page.getByRole("button", { name: /^Generate/ }).click();
  await expect(page.getByText("1 done", { exact: true })).toBeVisible({ timeout: 3_000 });
});

test("a new submission starts when a stopped queue contains only a failed job", async ({ page }) => {
  await page.goto("/?renderDelay=250");
  await page.getByLabel("Prompt").fill("fail preview while stopping");
  await page.getByRole("button", { name: /^Generate/ }).click();
  await expect(page.getByText("1 running", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Stop after current" }).click();
  await expect(page.getByText("1 failed", { exact: true })).toBeVisible({ timeout: 2_000 });
  await page.getByLabel("Prompt").fill("fresh after failed stop");
  await page.getByRole("button", { name: /^Generate/ }).click();
  await expect(page.getByText("1 done", { exact: true })).toBeVisible({ timeout: 3_000 });
  await expect(page.getByRole("button", { name: "Resume queue" })).toHaveCount(0);
});
