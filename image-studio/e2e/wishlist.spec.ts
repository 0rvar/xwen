import { expect, test } from "@playwright/test";

test("image deletion can be canceled and removes the gallery entry", async ({ page }) => {
  await page.goto("/");
  const tile = page.getByRole("button", { name: /Open A quiet architectural study/ });
  await tile.click();
  await page.getByRole("button", { name: "Delete image", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Delete image?" })).toBeVisible();
  await page.getByRole("button", { name: "Cancel", exact: true }).click();
  await expect(page.getByRole("dialog", { name: "Image preview" })).toBeVisible();
  await page.getByRole("button", { name: "Delete image", exact: true }).click();
  await page.getByRole("button", { name: "Delete permanently" }).click();
  await expect(tile).toHaveCount(0);
  await expect(page.getByText("Your renders will collect here")).toBeVisible();
  await expect(page.getByRole("button", { name: /^Delete session/ })).toHaveCount(1);
});

test("session deletion confirms all files and rotates current session", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText("Manage sessions", { exact: true })).toHaveCount(0);
  await expect(page.locator(".session-heading").getByRole("button", { name: /^Delete session/ })).toHaveCount(1);
  const oldSession = "20260908-142210-preview";
  await page.getByRole("button", { name: `Delete session ${oldSession}` }).click();
  await expect(page.getByText(/including all 1 images, YAML records and input snapshots/)).toBeVisible();
  await page.getByRole("button", { name: "Cancel", exact: true }).click();
  await expect(page.getByRole("button", { name: /Open A quiet architectural study/ })).toBeVisible();
  await page.getByRole("button", { name: `Delete session ${oldSession}` }).click();
  await page.getByRole("button", { name: "Delete permanently" }).click();
  await expect(page.getByText("Your renders will collect here")).toBeVisible();
  await expect(page.getByRole("button", { name: `Delete session ${oldSession}` })).toHaveCount(0);
  await page.getByRole("textbox", { name: "Prompt", exact: true }).fill("After deletion");
  await page.getByRole("button", { name: /^Generate/ }).click();
  await expect(page.getByText("1 done")).toBeVisible();
  await expect(page.getByRole("button", { name: /Open After deletion/ })).toBeVisible();
});

test("Generated drafts are editable and only replace the prompt on use", async ({ page }) => {
  await page.goto("/");
  const prompt = page.getByRole("textbox", { name: "Prompt", exact: true });
  await prompt.fill("Original idea");
  await page.getByRole("button", { name: "Draft image prompt" }).click();
  await expect(page.getByLabel("Idea or instructions")).toHaveValue("Original idea");
  await page.getByRole("button", { name: "Create draft" }).click();
  await expect(page.getByRole("textbox", { name: "Draft prompt", exact: true })).toHaveValue(/Original idea, soft evening light/);
  await expect(prompt).toHaveValue("Original idea");
  await page.getByRole("textbox", { name: "Draft prompt", exact: true }).fill("");
  await expect(page.getByRole("button", { name: "Use prompt", exact: true })).toBeDisabled();
  await page.getByRole("textbox", { name: "Draft prompt", exact: true }).fill("Edited generated prompt");
  await page.getByRole("button", { name: "Use prompt", exact: true }).click();
  await expect(prompt).toHaveValue("Edited generated prompt");
  await page.getByRole("button", { name: "Draft image prompt" }).click();
  await page.getByLabel("Idea or instructions").fill("fail prompt");
  await page.getByRole("button", { name: "Create draft" }).click();
  await expect(page.getByRole("alert")).toContainText("The language model is unavailable");
  await expect(page.getByRole("button", { name: "Use prompt", exact: true })).toBeDisabled();
  await page.getByRole("button", { name: "Close generator" }).click();
  await expect(prompt).toHaveValue("Edited generated prompt");
});
