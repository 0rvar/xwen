import { expect, test } from "@playwright/test";

test("first launch keeps setup visible after a canceled workspace picker", async ({ page }) => {
  await page.goto("/?firstLaunch=1&cancelPicker=1");
  await expect(page.getByRole("heading", { name: "Connect to xwen" })).toBeVisible();
  await page.getByLabel("Server URL").fill("http://127.0.0.1:5241/v1/");
  await page.getByRole("button", { name: "Save connection" }).click();
  await expect(page.getByRole("heading", { name: "Choose where your sessions live" })).toBeVisible();
  await page.getByRole("button", { name: "Choose workspace folder" }).click();
  await expect(page.getByRole("heading", { name: "Choose where your sessions live" })).toBeVisible();
  await page.getByRole("button", { name: "Choose workspace folder" }).click();
  await expect(page.getByLabel("Recent workspace")).toHaveValue("/Users/demo/Pictures/new-workspace");
});

test("checking a draft server does not save it", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Server settings" }).click();
  await page.getByLabel("Server URL").fill("http://draft.invalid:9999");
  await page.getByRole("button", { name: "Check server" }).click();
  await expect(page.getByText("Server is reachable.")).toBeVisible();
  await page.getByRole("button", { name: "Close settings" }).click();
  await page.getByRole("button", { name: "Server settings" }).click();
  await expect(page.getByLabel("Server URL")).toHaveValue("http://127.0.0.1:5241");
});

test("inpaint mask, independent control map, and server LoRA flow", async ({ page }) => {
  await page.goto("/?portrait=1");
  await page.screenshot({ path: "/tmp/xwen-image-studio-initial.png", fullPage: true });
  await page.getByRole("button", { name: "Inpaint" }).click();
  await page.getByRole("button", { name: "Choose image" }).click();
  await expect(page.getByText("source.png", { exact: true })).toBeVisible();
  await expect(page.getByLabel("Strength")).toHaveValue("1");
  await page.getByRole("button", { name: "Paint or import mask" }).click();
  const canvas = page.getByLabel("Mask painting canvas");
  await expect(canvas).toBeVisible();
  const box = await canvas.boundingBox();
  if (!box) throw new Error("Mask canvas has no bounds");
  const stageBox = await page.locator(".mask-stage").boundingBox();
  if (!stageBox) throw new Error("Mask stage has no bounds");
  expect(box.width).toBeLessThanOrEqual(stageBox.width + 1);
  expect(box.height).toBeLessThanOrEqual(stageBox.height + 1);
  expect(box.width / box.height).toBeCloseTo(.75, 1);
  await page.mouse.move(box.x + box.width * .35, box.y + box.height * .35);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * .65, box.y + box.height * .58, { steps: 5 });
  await page.mouse.up();
  await page.screenshot({ path: "/tmp/xwen-image-studio-mask.png", fullPage: true });
  await page.getByRole("button", { name: "Use mask" }).click();
  await expect(page.getByRole("button", { name: "Edit mask" })).toBeVisible();
  const maskPixels = await page.locator(".mask-ready-button img").evaluate(async (element: HTMLImageElement) => {
    await element.decode();
    const canvas = document.createElement("canvas");
    canvas.width = element.naturalWidth; canvas.height = element.naturalHeight;
    const context = canvas.getContext("2d")!;
    context.drawImage(element, 0, 0);
    return {
      center: [...context.getImageData(Math.floor(canvas.width * .5), Math.floor(canvas.height * .46), 1, 1).data],
      corner: [...context.getImageData(2, 2, 1, 1).data],
    };
  });
  expect(maskPixels.center[0]).toBeGreaterThan(240);
  expect(maskPixels.corner[0]).toBeLessThan(10);

  await page.getByText("ControlNet", { exact: true }).click();
  const control = page.locator("details").filter({ hasText: "ControlNet" });
  await control.getByRole("button", { name: "Choose image" }).click();
  await control.getByRole("button", { name: "Preview canny map" }).click();
  await control.getByLabel("Preprocessor").selectOption("none");
  await page.waitForTimeout(150);
  await expect(control.getByAltText("canny control map preview")).toHaveCount(0);
  await control.getByLabel("Preprocessor").selectOption("canny");
  await expect(control.getByRole("button", { name: "Preview canny map" })).toBeEnabled();
  await control.getByRole("button", { name: "Preview canny map" }).click();
  await expect(control.getByAltText("canny control map preview")).toBeVisible();
  await control.getByRole("button", { name: "Use this map" }).click();
  await expect(control.getByLabel("Preprocessor")).toHaveValue("none");

  await page.getByText("LoRAs", { exact: true }).click();
  const lora = page.locator("details").filter({ hasText: "LoRAs" });
  await lora.getByRole("button", { name: "Refresh server LoRAs" }).click();
  await lora.getByLabel("Available LoRA").selectOption("/models/ceramic-light.safetensors");
  await lora.getByRole("button", { name: "Add" }).click();
  await expect(lora.getByText("ceramic-light.safetensors")).toBeVisible();
  await page.getByLabel("Prompt").fill("Controlled inpaint test");
  await page.getByRole("button", { name: /^Generate/ }).click();
  await expect(page.getByText("1 done")).toBeVisible({ timeout: 3_000 });
  const request = await page.evaluate(() => (window as typeof window & { __XWEN_PREVIEW_REQUESTS__?: Array<Record<string, unknown>> }).__XWEN_PREVIEW_REQUESTS__?.[0]);
  expect(request?.mask).toMatch(/^data:image\/png/);
  expect((request?.control as { preprocess?: string }).preprocess).toBe("none");
  expect((request?.loras as Array<{ name: string }>)[0]?.name).toBe("/models/ceramic-light.safetensors");
});

test("matrix preview runs its exact jobs sequentially and can pause", async ({ page }) => {
  await page.goto("/");
  await page.getByLabel("Prompt").fill("Base prompt");
  await page.getByText("Batch variations", { exact: true }).click();
  const batch = page.locator("details").filter({ hasText: "Batch variations" });
  await batch.getByRole("button", { name: "Add axis" }).click();
  await batch.getByLabel("Batch values 1").fill("Clay vessel\nGlass vessel");
  await batch.getByRole("button", { name: "Add axis" }).click();
  await expect(batch.getByLabel("Batch parameter 2")).toHaveValue("seed");
  await batch.getByLabel("Batch values 2").fill("10, 20");
  await batch.getByRole("button", { name: "Preview batch" }).click();
  await expect(batch.getByText("4 images")).toBeVisible();
  await batch.getByRole("button", { name: "Queue exact preview" }).click();
  await page.getByRole("button", { name: "Stop after current" }).click();
  await expect(page.getByRole("button", { name: "Resume queue" })).toBeVisible();
  await expect(page.getByText(/waiting/).first()).toBeVisible();
  await page.getByRole("button", { name: "Resume queue" }).click();
  await expect(page.getByText("4 done")).toBeVisible({ timeout: 5_000 });
  await expect(page.getByText("Clay vessel", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("Glass vessel", { exact: true }).first()).toBeVisible();
  await page.screenshot({ path: "/tmp/xwen-image-studio-main.png", fullPage: true });
});

test("paired validation, failed render retry, metadata restore, and workspace switch", async ({ page }) => {
  await page.goto("/");
  const tile = page.getByRole("button", { name: /Open A quiet architectural study/ });
  await tile.click();
  await page.getByRole("button", { name: "Restore settings" }).click();
  await expect(page.getByRole("button", { name: "Image", exact: true })).toHaveClass(/selected/);
  await expect(page.getByText("source.png", { exact: true })).toBeVisible();

  await page.getByLabel("Prompt").fill("fail preview");
  await page.getByRole("button", { name: /^Generate/ }).click();
  await expect(page.getByText("Preview render failed as requested.")).toBeVisible();
  await page.getByRole("button", { name: "Retry" }).click();
  await expect(page.getByText("1 done")).toBeVisible({ timeout: 3_000 });

  await page.getByText("Batch variations", { exact: true }).click();
  const batch = page.locator("details").filter({ hasText: "Batch variations" });
  await batch.getByRole("button", { name: "Paired" }).click();
  await batch.getByRole("button", { name: "Add axis" }).click();
  await batch.getByLabel("Batch values 1").fill("One\nTwo");
  await batch.getByRole("button", { name: "Add axis" }).click();
  await batch.getByLabel("Batch values 2").fill("1,2,3");
  await batch.getByRole("button", { name: "Preview batch" }).click();
  await expect(batch.getByText(/Paired axes must have the same number/)).toBeVisible();

  await page.getByRole("button", { name: "Open…" }).click();
  await expect(page.getByLabel("Recent workspace")).toHaveValue("/Users/demo/Pictures/new-workspace");
  await expect(page.getByText("Your renders will collect here")).toBeVisible();
});
