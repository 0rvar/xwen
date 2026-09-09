import { expect, test, type Page } from "@playwright/test";
import type { NativeBridge } from "../src/bridge";
import type { BatchDraft } from "../src/batchDraft";

declare global {
  interface Window {
    __chatTurnTest: {
      calls: unknown[][];
      events: string[];
      drafts: BatchDraft[];
      renders: string[];
      finish(reason?: "stop" | "length"): void;
    };
    __workspaceTransitionTest: {
      directoryPending: boolean; workspacePending: boolean; chats: number;
      sessionId: string; batchSessions: string[];
      chooseDirectory(path?: string | null): void; selectWorkspace(): void;
    };
  }
}

async function prepareTurn(page: Page) {
  await page.goto("/");
  await page.getByRole("button", { name: "Chat with image assistant", exact: true }).click();
  await page.evaluate(async () => {
    const modulePath = "/src/bridge.ts";
    const { bridge } = await import(modulePath) as { bridge: NativeBridge };
    const createBatch = bridge.createBatch;
    const renderBatchJob = bridge.renderBatchJob;
    let finish: (reason: "stop" | "length") => void = () => { throw new Error("No final reply is pending."); };
    const state: Window["__chatTurnTest"] = {
      calls: [], events: [], drafts: [], renders: [], finish: (reason = "stop") => finish(reason),
    };
    window.__chatTurnTest = state;
    bridge.createBatch = async (sessionId, draft) => {
      state.events.push("createBatch"); state.drafts.push(structuredClone(draft));
      return createBatch(sessionId, draft);
    };
    bridge.renderBatchJob = async (sessionId, batchId, jobId) => {
      state.events.push("render"); state.renders.push(jobId);
      return renderBatchJob(sessionId, batchId, jobId);
    };
    bridge.chat = async (messages) => {
      state.calls.push(structuredClone(messages));
      const round = state.calls.length;
      state.events.push(`chat-${round}`);
      if (round === 1 || round === 2) {
        const jobs = round === 1
          ? [{ prompt: "Terracotta pavilion", seed: 101, n: 2 }, { prompt: "Blue courtyard", seed: 201 }]
          : [{ prompt: "Green conservatory", seed: 301 }];
        return { choices: [{ finish_reason: "tool_calls", message: { role: "assistant", tool_calls: [{ id: `call-${round}`, type: "function", function: { name: "queue_txt2img", arguments: JSON.stringify({ jobs }) } }] } }] };
      }
      if (round === 3) return new Promise((resolve) => {
        finish = (reason) => {
          state.events.push("final");
          resolve({ choices: [{ finish_reason: reason, message: { role: "assistant", content: "Prepared four images across three garden ideas." } }] });
        };
      });
      return { choices: [{ finish_reason: "stop", message: { role: "assistant", content: "We can try a smaller batch next." } }] };
    };
  });
  await page.getByRole("textbox", { name: "Message", exact: true }).fill("Make four images across three garden ideas");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await expect.poll(() => page.evaluate(() => window.__chatTurnTest.calls.length)).toBe(3);
  await expect(page.getByText("4 images staged. Finishing the complete batch…", { exact: true })).toBeVisible();
}

async function expectNoSubmission(page: Page) {
  expect(await page.evaluate(() => ({ drafts: window.__chatTurnTest.drafts.length, renders: window.__chatTurnTest.renders.length }))).toEqual({ drafts: 0, renders: 0 });
}

test("a whole multi-round chat batch waits for the final reply before saving and rendering", async ({ page }) => {
  await prepareTurn(page);
  await expectNoSubmission(page);
  await expect(page.getByRole("button", { name: "New session", exact: true })).toBeDisabled();
  await expect(page.getByRole("combobox", { name: "Recent workspace", exact: true })).toBeDisabled();
  await expect(page.getByRole("button", { name: "Server settings", exact: true })).toBeDisabled();
  await expect(page.locator("#image-assistant").getByRole("button", { name: "Clear", exact: true })).toBeDisabled();
  await page.getByRole("textbox", { name: "Prompt", exact: true }).fill("Controls remain usable during chat");
  await expectNoSubmission(page);

  await page.evaluate(() => window.__chatTurnTest.finish());
  await expect(page.getByText("4 images queued.", { exact: true })).toBeVisible();
  await expect(page.getByText("4 done", { exact: true })).toBeVisible();
  const result = await page.evaluate(() => ({ drafts: window.__chatTurnTest.drafts, events: window.__chatTurnTest.events, renders: window.__chatTurnTest.renders }));
  expect(result.drafts).toHaveLength(1);
  expect(result.drafts[0]!.jobs.map((job) => [job.request.prompt, job.request.seed, job.request.n])).toEqual([
    ["Terracotta pavilion", 101, 1], ["Terracotta pavilion", 102, 1], ["Blue courtyard", 201, 1], ["Green conservatory", 301, 1],
  ]);
  expect(result.events).toEqual(["chat-1", "chat-2", "chat-3", "final", "createBatch", "render", "render", "render", "render"]);
  expect(result.renders).toHaveLength(4);
  await expect(page.getByRole("button", { name: "New session", exact: true })).toBeEnabled();
});

test("pausing while the assistant finishes keeps its complete batch waiting until resume", async ({ page }) => {
  await prepareTurn(page);
  await page.getByRole("button", { name: "Pause queue", exact: true }).click();
  await expectNoSubmission(page);
  await page.evaluate(() => window.__chatTurnTest.finish());
  await expect(page.getByText("4 images queued.", { exact: true })).toBeVisible();
  await expect(page.getByText("4 waiting", { exact: true })).toBeVisible();
  expect(await page.evaluate(() => ({ drafts: window.__chatTurnTest.drafts.length, renders: window.__chatTurnTest.renders.length }))).toEqual({ drafts: 1, renders: 0 });
  await page.getByRole("button", { name: "Resume queue", exact: true }).click();
  await expect(page.getByText("4 done", { exact: true })).toBeVisible();
  expect(await page.evaluate(() => window.__chatTurnTest.drafts.length)).toBe(1);
});

test("a truncated final reply discards staged jobs and leaves no dangling tools in the next turn", async ({ page }) => {
  await prepareTurn(page);
  await page.evaluate(() => window.__chatTurnTest.finish("length"));
  await expect(page.locator("#image-assistant").getByRole("alert")).toContainText("output token limit");
  await expectNoSubmission(page);
  await expect(page.getByRole("button", { name: "New session", exact: true })).toBeEnabled();
  await page.getByRole("textbox", { name: "Message", exact: true }).fill("Let's discuss one idea instead");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await expect(page.getByText("We can try a smaller batch next.", { exact: true })).toBeVisible();
  const nextHistory = await page.evaluate(() => window.__chatTurnTest.calls[3] as Array<{ role: string; tool_calls?: unknown }>);
  expect(nextHistory.map((message) => message.role)).toEqual(["system", "user", "user"]);
  expect(nextHistory.every((message) => message.tool_calls === undefined)).toBe(true);
  await expectNoSubmission(page);
});

test("chat waits for the workspace picker and selection, then queues into the new session", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Chat with image assistant", exact: true }).click();
  await page.getByRole("textbox", { name: "Message", exact: true }).fill("Make an image in the new workspace");
  await page.evaluate(async () => {
    const modulePath = "/src/bridge.ts";
    const { bridge } = await import(modulePath) as { bridge: NativeBridge };
    const selectWorkspace = bridge.selectWorkspace;
    const createBatch = bridge.createBatch;
    const state: Window["__workspaceTransitionTest"] = {
      directoryPending: false, workspacePending: false, chats: 0, sessionId: "", batchSessions: [],
      chooseDirectory: () => {}, selectWorkspace: () => {},
    };
    window.__workspaceTransitionTest = state;
    bridge.chooseDirectory = async () => new Promise((resolve) => {
      state.directoryPending = true;
      state.chooseDirectory = (path = "/Users/demo/Pictures/new-chat-workspace") => { state.directoryPending = false; resolve(path); };
    });
    bridge.selectWorkspace = async (path) => {
      state.workspacePending = true;
      await new Promise<void>((resolve) => { state.selectWorkspace = resolve; });
      const workspace = await selectWorkspace(path);
      state.sessionId = workspace.session_id;
      state.workspacePending = false;
      return workspace;
    };
    bridge.createBatch = async (sessionId, draft) => {
      state.batchSessions.push(sessionId);
      return createBatch(sessionId, draft);
    };
    bridge.chat = async () => {
      state.chats += 1;
      return state.chats === 1
        ? { choices: [{ finish_reason: "tool_calls", message: { role: "assistant", tool_calls: [{ id: "new-workspace-job", type: "function", function: { name: "queue_txt2img", arguments: JSON.stringify({ jobs: [{ prompt: "A fresh workspace", seed: 402 }] }) } }] } }] }
        : { choices: [{ finish_reason: "stop", message: { role: "assistant", content: "Prepared one image in your new workspace." } }] };
    };
  });
  await page.getByRole("button", { name: "Open…", exact: true }).click();
  await expect.poll(() => page.evaluate(() => window.__workspaceTransitionTest.directoryPending)).toBe(true);
  await expect(page.getByRole("button", { name: "Send", exact: true })).toBeDisabled();
  expect(await page.evaluate(() => window.__workspaceTransitionTest.chats)).toBe(0);
  await page.evaluate(() => window.__workspaceTransitionTest.chooseDirectory(null));
  await expect(page.getByRole("button", { name: "Send", exact: true })).toBeEnabled();
  await page.getByRole("button", { name: "Open…", exact: true }).click();
  await expect.poll(() => page.evaluate(() => window.__workspaceTransitionTest.directoryPending)).toBe(true);
  await expect(page.getByRole("button", { name: "Send", exact: true })).toBeDisabled();
  await page.evaluate(() => window.__workspaceTransitionTest.chooseDirectory());
  await expect.poll(() => page.evaluate(() => window.__workspaceTransitionTest.workspacePending)).toBe(true);
  await expect(page.getByRole("button", { name: "Send", exact: true })).toBeDisabled();
  expect(await page.evaluate(() => window.__workspaceTransitionTest.chats)).toBe(0);
  await page.evaluate(() => window.__workspaceTransitionTest.selectWorkspace());
  await expect(page.getByRole("button", { name: "Send", exact: true })).toBeEnabled();
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await expect(page.getByText("1 image queued.", { exact: true })).toBeVisible();
  await expect(page.getByText("1 done", { exact: true })).toBeVisible();
  const result = await page.evaluate(() => ({ sessions: window.__workspaceTransitionTest.batchSessions, sessionId: window.__workspaceTransitionTest.sessionId }));
  expect(result.sessionId).not.toBe("20260908-142210-preview");
  expect(result.sessions).toEqual([result.sessionId]);
});
