export interface SchedulerState { paused: boolean; held: boolean }

/** A chat turn holds dispatch until its last reply, after any active render drains. */
export function createRenderScheduler(onChange: (state: SchedulerState) => void = () => {}) {
  let paused = false;
  let holds = 0;
  let running = false;
  const idleWaiters = new Set<() => void>();
  const changed = () => onChange({ paused, held: holds > 0 });
  const setPaused = (next: boolean) => { paused = next; changed(); };

  return {
    setPaused,
    prepareRun(startsNewRun: boolean) {
      if (startsNewRun && holds === 0) setPaused(false);
    },
    tryStart() {
      if (paused || holds > 0 || running) return false;
      running = true;
      return true;
    },
    finish() {
      running = false;
      for (const resolve of idleWaiters) resolve();
      idleWaiters.clear();
    },
    async acquireTurn(): Promise<() => void> {
      holds += 1;
      changed();
      if (running) await new Promise<void>((resolve) => idleWaiters.add(resolve));
      let released = false;
      return () => {
        if (released) return;
        released = true;
        holds -= 1;
        changed();
      };
    },
  };
}
