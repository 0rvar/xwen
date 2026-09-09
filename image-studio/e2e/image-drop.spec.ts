import { expect, test, type Page } from "@playwright/test";

async function imageTransfer(page: Page, name: string, width: number, height: number) {
  return page.evaluateHandle(async ({ name, width, height }) => {
    const canvas = document.createElement("canvas");
    canvas.width = width; canvas.height = height;
    const context = canvas.getContext("2d")!;
    context.fillStyle = "#bd684d"; context.fillRect(0, 0, width, height);
    const blob = await new Promise<Blob>((resolve, reject) => canvas.toBlob((value) => value ? resolve(value) : reject(new Error("PNG encoding failed")), "image/png"));
    const transfer = new DataTransfer();
    transfer.items.add(new File([blob], name, { type: "image/png" }));
    return transfer;
  }, { name, width, height });
}

async function dropOn(page: Page, selector: string, dataTransfer: Awaited<ReturnType<typeof imageTransfer>>) {
  const target = page.locator(selector);
  const bounds = await target.boundingBox();
  if (!bounds) throw new Error(`No bounds for ${selector}`);
  await target.dispatchEvent("dragenter", { dataTransfer, clientX: bounds.x + bounds.width / 2, clientY: bounds.y + bounds.height / 2 });
  await expect(target).toHaveClass(/drop-active/);
  await target.dispatchEvent("drop", { dataTransfer, clientX: bounds.x + bounds.width / 2, clientY: bounds.y + bounds.height / 2 });
}

test("source and control accept drops and populated inputs accept replacement", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Inpaint", exact: true }).click();
  await page.getByRole("textbox", { name: "Prompt", exact: true }).fill("Keep this edit during image decoding");
  await dropOn(page, "[data-image-drop=source]", await imageTransfer(page, "first-source.png", 640, 480));
  await expect(page.getByText("first-source.png", { exact: true })).toBeVisible();
  await expect(page.getByRole("textbox", { name: "Prompt", exact: true })).toHaveValue("Keep this edit during image decoding");
  await page.getByRole("button", { name: "Paint or import mask" }).click();
  await page.getByRole("button", { name: "Use mask" }).click();
  await expect(page.getByRole("button", { name: "Edit mask" })).toBeVisible();

  await dropOn(page, "[data-image-drop=source]", await imageTransfer(page, "replacement.png", 320, 512));
  await expect(page.getByText("replacement.png", { exact: true })).toBeVisible();
  await expect(page.getByText("first-source.png", { exact: true })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Paint or import mask" })).toBeVisible();
  await expect(page.getByRole("spinbutton", { name: "Width" })).toHaveValue("320");
  await expect(page.getByRole("spinbutton", { name: "Height" })).toHaveValue("512");

  await page.getByText("ControlNet", { exact: true }).click();
  await dropOn(page, "[data-image-drop=control]", await imageTransfer(page, "control-map.jpg", 512, 512));
  await expect(page.getByText("control-map.jpg", { exact: true })).toBeVisible();
  await expect(page.getByText("replacement.png", { exact: true })).toBeVisible();
});

test("multi-file and non-image drops are rejected without replacing the image", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Image", exact: true }).click();
  await dropOn(page, "[data-image-drop=source]", await imageTransfer(page, "accepted.png", 512, 512));
  await expect(page.getByText("accepted.png", { exact: true })).toBeVisible();
  const target = page.locator("[data-image-drop=source]");
  const bounds = await target.boundingBox();
  if (!bounds) throw new Error("No source bounds");
  const multiple = await page.evaluateHandle(async () => {
    const transfer = new DataTransfer();
    for (const name of ["one.png", "two.png"]) {
      const canvas = document.createElement("canvas"); canvas.width = 16; canvas.height = 16;
      const blob = await new Promise<Blob>((resolve) => canvas.toBlob((value) => resolve(value!), "image/png"));
      transfer.items.add(new File([blob], name, { type: "image/png" }));
    }
    return transfer;
  });
  await target.dispatchEvent("drop", { dataTransfer: multiple, clientX: bounds.x + 10, clientY: bounds.y + 10 });
  await expect(page.getByText("Choose exactly one PNG or JPEG image.")).toBeVisible();
  await expect(page.getByText("accepted.png", { exact: true })).toBeVisible();

  const text = await page.evaluateHandle(() => {
    const transfer = new DataTransfer(); transfer.items.add(new File(["not an image"], "notes.txt", { type: "text/plain" })); return transfer;
  });
  await target.dispatchEvent("drop", { dataTransfer: text, clientX: bounds.x + 10, clientY: bounds.y + 10 });
  await expect(page.getByText("Choose exactly one PNG or JPEG image.")).toBeVisible();
  await expect(page.getByText("accepted.png", { exact: true })).toBeVisible();
});

test("modal overlays block underlying drop targets", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Image", exact: true }).click();
  await dropOn(page, "[data-image-drop=source]", await imageTransfer(page, "before-modal.png", 512, 512));
  await expect(page.getByText("before-modal.png", { exact: true })).toBeVisible();
  const target = page.locator("[data-image-drop=source]");
  const bounds = await target.boundingBox();
  if (!bounds) throw new Error("No source bounds");
  await page.getByRole("button", { name: "Draft image prompt" }).click();
  await expect(page.getByRole("dialog", { name: "Prompt generator" })).toBeVisible();
  const transfer = await imageTransfer(page, "blocked-by-modal.png", 512, 512);
  await target.dispatchEvent("drop", { dataTransfer: transfer, clientX: bounds.x + bounds.width / 2, clientY: bounds.y + bounds.height / 2 });
  await page.waitForTimeout(100);
  await expect(page.getByText("before-modal.png", { exact: true })).toBeVisible();
  await expect(page.getByText("blocked-by-modal.png", { exact: true })).toHaveCount(0);
});

test("chooser import preserves settings edited while its image is loading", async ({ page }) => {
  await page.goto("/?readDelay=300");
  await page.getByRole("button", { name: "Image", exact: true }).click();
  await page.getByRole("button", { name: /Choose image/ }).click();
  await page.getByRole("textbox", { name: "Prompt", exact: true }).fill("Edited after choosing");
  await page.getByRole("spinbutton", { name: "Steps" }).fill("12");
  await expect(page.getByText("source.png", { exact: true })).toBeVisible();
  await expect(page.getByRole("textbox", { name: "Prompt", exact: true })).toHaveValue("Edited after choosing");
  await expect(page.getByRole("spinbutton", { name: "Steps" })).toHaveValue("12");
});
