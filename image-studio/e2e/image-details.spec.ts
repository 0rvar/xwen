import { expect, test } from "@playwright/test";

test("existing metadata shows recorded edit parameters without inventing missing values", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: /Open A quiet architectural study/ }).click();
  const details = page.getByRole("region", { name: "Generation parameters" });
  await expect(details).toBeVisible();
  await expect(details).toContainText("ModeImage to image");
  await expect(details).toContainText("Steps8");
  await expect(details).toContainText("Strength0.6");
  await expect(details).toContainText("ControlNetNone");
  await expect(details).toContainText("ModelNot recorded");
  await expect(details).toContainText("LoRAsNone");
});

test("zero-valued inpainting, control and LoRA parameters stay readable", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Inpaint", exact: true }).click();
  await page.getByRole("button", { name: "Choose image" }).click();
  await page.getByRole("button", { name: "Paint or import mask" }).click();
  await page.getByRole("button", { name: "Use mask" }).click();
  await page.getByRole("spinbutton", { name: "Strength" }).fill("0");
  await page.getByRole("spinbutton", { name: "Mask blur" }).fill("4");

  await page.getByText("ControlNet", { exact: true }).click();
  const control = page.locator("details").filter({ hasText: "ControlNet" });
  await control.getByRole("button", { name: "Choose image" }).click();
  await control.getByLabel("Preprocessor").selectOption("none");
  await control.getByRole("spinbutton", { name: "Scale" }).fill("0");
  await control.getByRole("spinbutton", { name: "Start" }).fill("0.1");
  await control.getByRole("spinbutton", { name: "End" }).fill("0.9");

  await page.getByText("LoRAs", { exact: true }).click();
  const loras = page.locator("details").filter({ hasText: "LoRAs" });
  await loras.getByRole("button", { name: "Refresh server LoRAs" }).click();
  await loras.getByLabel("Available LoRA").selectOption("/models/ceramic-light.safetensors");
  await loras.getByRole("button", { name: "Add" }).click();
  await loras.getByRole("spinbutton", { name: "LoRA 1 weight" }).fill("0");

  await page.getByRole("textbox", { name: "Prompt", exact: true }).fill("Readable parameters test");
  await page.getByRole("button", { name: /^Generate/ }).click();
  await expect(page.getByText("1 done", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: /Open Readable parameters test/ }).click();
  const details = page.getByRole("region", { name: "Generation parameters" });
  await expect(details).toContainText("ModeInpainting");
  await expect(details).toContainText("Steps8");
  await expect(details).toContainText("Strength0");
  await expect(details).toContainText("Mask blur4");
  await expect(details).toContainText("ControlNetPrepared map");
  await expect(details).toContainText("Control scale0");
  await expect(details).toContainText("Control start0.1");
  await expect(details).toContainText("Control end0.9");
  await expect(details).toContainText("ceramic-light.safetensors · 0");
});
