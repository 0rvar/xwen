import { describe, expect, test } from "bun:test";
import { createRenderScheduler, type SchedulerState } from "./renderScheduler";

describe("render scheduling around assistant turns", () => {
  test("takes the hold immediately and starts chat only after an active render settles", async () => {
    const states: SchedulerState[] = [];
    const scheduler = createRenderScheduler((state) => states.push(state));
    expect(scheduler.tryStart()).toBe(true);
    let chatStarted = false;
    const turn = scheduler.acquireTurn().then((release) => { chatStarted = true; return release; });
    expect(states.at(-1)?.held).toBe(true);
    await Promise.resolve();
    expect(chatStarted).toBe(false);
    expect(scheduler.tryStart()).toBe(false);
    scheduler.finish();
    const release = await turn;
    expect(chatStarted).toBe(true);
    expect(scheduler.tryStart()).toBe(false);
    release();
    expect(scheduler.tryStart()).toBe(true);
  });

  test("pending and newly submitted jobs stay blocked through multiple tool rounds", async () => {
    const scheduler = createRenderScheduler();
    const turn = scheduler.acquireTurn();
    expect(scheduler.tryStart()).toBe(false);
    const release = await turn;
    for (let round = 0; round < 3; round += 1) {
      scheduler.prepareRun(round === 0);
      await Promise.resolve();
      expect(scheduler.tryStart()).toBe(false);
    }
    release();
    expect(scheduler.tryStart()).toBe(true);
    expect(scheduler.tryStart()).toBe(false);
    scheduler.finish();
    expect(scheduler.tryStart()).toBe(true);
  });

  test("submission and release preserve a pause requested before or during chat", async () => {
    for (const pauseBeforeTurn of [true, false]) {
      const scheduler = createRenderScheduler();
      if (pauseBeforeTurn) scheduler.setPaused(true);
      const release = await scheduler.acquireTurn();
      if (!pauseBeforeTurn) scheduler.setPaused(true);
      scheduler.prepareRun(true);
      release();
      expect(scheduler.tryStart()).toBe(false);
      scheduler.setPaused(false);
      expect(scheduler.tryStart()).toBe(true);
    }
  });

  test("an ordinary new submission retains the existing drained-queue resume behavior", () => {
    const scheduler = createRenderScheduler();
    scheduler.setPaused(true);
    scheduler.prepareRun(false);
    expect(scheduler.tryStart()).toBe(false);
    scheduler.prepareRun(true);
    expect(scheduler.tryStart()).toBe(true);
  });

  test("resuming a paused queue during chat waits for release before dispatch", async () => {
    const scheduler = createRenderScheduler();
    scheduler.setPaused(true);
    const release = await scheduler.acquireTurn();
    scheduler.setPaused(false);
    expect(scheduler.tryStart()).toBe(false);
    release();
    expect(scheduler.tryStart()).toBe(true);
  });

  test("failed turns release dispatch in finally and duplicate releases cannot release another hold", async () => {
    const scheduler = createRenderScheduler();
    const otherRelease = await scheduler.acquireTurn();
    const release = await scheduler.acquireTurn();
    const failedTurn = async () => {
      try { throw new Error("Chat request failed"); }
      finally { release(); }
    };
    await expect(failedTurn()).rejects.toThrow("Chat request failed");
    release();
    expect(scheduler.tryStart()).toBe(false);
    otherRelease();
    expect(scheduler.tryStart()).toBe(true);
  });
});
