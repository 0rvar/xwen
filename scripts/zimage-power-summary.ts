#!/usr/bin/env bun
// Bin a powermetrics trace by the denoising steps of a Z-Image run.
//
// Input is the directory scripts/zimage-power.sh writes: `powermetrics.log`
// (samplers gpu_power,cpu_power,thermal) and `xwen.log` (each xwen line prefixed
// with epoch seconds). Output is one row per step with the mean GPU power, the
// mean and minimum GPU frequency and the thermal pressure levels seen in that
// step's window, then the same for the VAE decode, then the whole trace as a
// timeline when --timeline is passed.
//
//   bun scripts/zimage-power-summary.ts /tmp/zimage-power-<stamp> [--timeline]

import { readFileSync } from "node:fs";
import { join } from "node:path";

type Sample = {
  t: number; // epoch seconds at the END of the sample window
  dt: number; // window length in seconds
  gpuMw?: number;
  gpuMhz?: number;
  gpuActivePct?: number;
  gpuSwTop?: string;
  combinedMw?: number;
  pressure?: string;
};

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

// `Tue Sep  8 12:00:00 2026 +0200` -> epoch seconds. Date.parse is not reliable
// on this shape, so it is taken apart by hand.
function parseStamp(s: string): number | undefined {
  const m = s.match(/\w{3}\s+(\w{3})\s+(\d+)\s+(\d+):(\d+):(\d+)\s+(\d{4})\s+([+-])(\d{2})(\d{2})/);
  if (!m) return undefined;
  const month = MONTHS.indexOf(m[1]);
  if (month < 0) return undefined;
  const utc = Date.UTC(+m[6], month, +m[2], +m[3], +m[4], +m[5]);
  const offset = (+m[8] * 60 + +m[9]) * 60 * 1000 * (m[7] === "-" ? -1 : 1);
  return (utc - offset) / 1000;
}

function parsePowermetrics(text: string): Sample[] {
  const samples: Sample[] = [];
  let cur: Sample | undefined;
  for (const line of text.split("\n")) {
    const head = line.match(/^\*\*\* Sampled system activity \((.+?)\) \(([\d.]+)ms elapsed\)/);
    if (head) {
      const t = parseStamp(head[1]);
      const dt = +head[2] / 1000;
      if (t !== undefined) {
        cur = { t, dt };
        samples.push(cur);
      } else {
        cur = undefined;
      }
      continue;
    }
    if (!cur) continue;
    let m: RegExpMatchArray | null;
    if ((m = line.match(/GPU (?:HW )?active frequency:\s*([\d.]+)\s*MHz/i))) cur.gpuMhz = +m[1];
    else if ((m = line.match(/GPU HW active residency:\s*([\d.]+)%/i))) cur.gpuActivePct = +m[1];
    else if ((m = line.match(/GPU SW requested state: \((.*)\)/i))) cur.gpuSwTop = topRequestedState(m[1]);
    else if ((m = line.match(/^GPU Power:\s*([\d.]+)\s*mW/i))) cur.gpuMw = +m[1];
    else if ((m = line.match(/Combined Power.*?:\s*([\d.]+)\s*mW/i))) cur.combinedMw = +m[1];
    else if ((m = line.match(/Current pressure level:\s*(\w+)/i))) cur.pressure = m[1];
  }
  // The header stamp has one-second resolution, so a 250 ms sample grid read
  // off it bunches four samples on one second. The elapsed values are exact:
  // accumulate them into a continuous clock and anchor that clock to the
  // stamps, which are the true times truncated to the second, so on average
  // half a second early.
  if (samples.length > 1) {
    let acc = 0;
    const accumulated = samples.map((s) => (acc += s.dt));
    const offsets = samples.map((s, i) => s.t - accumulated[i]);
    const anchor = offsets.reduce((a, b) => a + b, 0) / offsets.length + 0.5;
    samples.forEach((s, i) => (s.t = accumulated[i] + anchor));
  }
  return samples;
}

type Window = { name: string; start: number; end: number; dur: number };

// The CLI prints every step line and the VAE line only after the whole render
// returns, so their timestamps all fall within a few milliseconds of each other
// and say nothing about when each step ran. What they carry is each stage's
// exact duration, and the render starts right after the seed line, so the
// windows are laid end to end from that line forward.
function parseWindows(text: string): Window[] {
  const windows: Window[] = [];
  let cursor: number | undefined;
  for (const raw of text.split("\n")) {
    const m = raw.match(/^([\d.]+)\s+(.*)$/);
    if (!m) continue;
    const t = +m[1];
    const line = m[2];
    let s: RegExpMatchArray | null;
    if ((s = line.match(/xwen: step (\d+)\/(\d+) ([\d.]+)s/))) {
      const dur = +s[3];
      const start = cursor ?? t - dur;
      windows.push({ name: `step ${s[1]}/${s[2]}`, start, end: start + dur, dur });
      cursor = start + dur;
    } else if ((s = line.match(/xwen: VAE decode ([\d.]+)s/))) {
      const dur = +s[1];
      const start = cursor ?? t - dur;
      windows.push({ name: "VAE decode", start, end: start + dur, dur });
      cursor = start + dur;
    } else if ((s = line.match(/xwen: (text encoder|transformer and VAE) loaded in ([\d.]+)s/))) {
      const dur = +s[2];
      windows.push({ name: `${s[1]} load`, start: t - dur, end: t, dur });
    } else if (line.match(/xwen: seed \d+|xwen: latents from/)) {
      cursor = t;
    }
  }
  return windows;
}

function mean(xs: number[]): number | undefined {
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : undefined;
}

function fmt(x: number | undefined, digits = 0): string {
  return x === undefined ? "-" : x.toFixed(digits);
}

const dir = process.argv[2];
if (!dir) {
  console.error("usage: bun scripts/zimage-power-summary.ts <dir> [--timeline]");
  process.exit(2);
}
const timeline = process.argv.includes("--timeline");

const samples = parsePowermetrics(readFileSync(join(dir, "powermetrics.log"), "utf8"));
const windows = parseWindows(readFileSync(join(dir, "xwen.log"), "utf8"));

if (samples.length === 0) {
  console.error("no powermetrics samples parsed; the sampler names or the header format changed");
  process.exit(1);
}
if (windows.length === 0) {
  console.error("no xwen step lines parsed from xwen.log");
  process.exit(1);
}

// A sample belongs to a window when its midpoint falls inside it.
function inWindow(w: Window): Sample[] {
  return samples.filter((s) => {
    const mid = s.t - s.dt / 2;
    return mid >= w.start && mid < w.end;
  });
}

const rows = windows.map((w) => {
  const ss = inWindow(w);
  const mw = ss.map((s) => s.gpuMw).filter((x): x is number => x !== undefined);
  const mhz = ss.map((s) => s.gpuMhz).filter((x): x is number => x !== undefined);
  const comb = ss.map((s) => s.combinedMw).filter((x): x is number => x !== undefined);
  const active = ss.map((s) => s.gpuActivePct).filter((x): x is number => x !== undefined);
  const swTop = [...new Set(ss.map((s) => s.gpuSwTop).filter(Boolean))].join(" ") || "-";
  const pressure = [...new Set(ss.map((s) => s.pressure).filter(Boolean))].join(",") || "-";
  return {
    window: w.name,
    "dur s": w.dur.toFixed(2),
    samples: ss.length,
    "GPU W": fmt(mean(mw) && mean(mw)! / 1000, 1),
    "GPU W max": mw.length ? (Math.max(...mw) / 1000).toFixed(1) : "-",
    "GPU MHz": fmt(mean(mhz)),
    "MHz min": mhz.length ? String(Math.min(...mhz)) : "-",
    "MHz max": mhz.length ? String(Math.max(...mhz)) : "-",
    "active %": fmt(mean(active)),
    "SW asks": swTop,
    "CPU+GPU+ANE W": fmt(mean(comb) && mean(comb)! / 1000, 1),
    pressure,
  };
});

console.log(`${samples.length} powermetrics samples, ${windows.length} xwen windows`);
console.table(rows);

const stepRows = rows.filter((r) => r.window.startsWith("step "));
if (stepRows.length >= 2) {
  const first = stepRows[0];
  const last = stepRows[stepRows.length - 1];
  console.log(
    `first step ${first["dur s"]} s at ${first["GPU W"]} W / ${first["GPU MHz"]} MHz; ` +
      `last step ${last["dur s"]} s at ${last["GPU W"]} W / ${last["GPU MHz"]} MHz`,
  );
  console.log(
    "reading: flat power with falling MHz as the step lengthens is a power cap; " +
      "falling power AND MHz with a pressure level above Nominal is thermal; " +
      "flat power and flat MHz with a lengthening step is neither, look elsewhere",
  );
}

if (timeline) {
  const t0 = samples[0].t - samples[0].dt;
  console.log("\ntimeline (t since first sample, s)");
  for (const s of samples) {
    console.log(
      `${(s.t - t0).toFixed(2).padStart(7)}  ${fmt(s.gpuMw && s.gpuMw / 1000, 1).padStart(5)} W  ` +
        `${fmt(s.gpuMhz).padStart(5)} MHz  ${s.pressure ?? "-"}`,
    );
  }
}

// `P1 : 100% P2 : 0% ...` -> the highest P-state the driver asked for with a
// nonzero share. Read beside the HW frequency: a high request
// met by a low frequency is the hardware refusing, a low request is the driver
// choosing.
function topRequestedState(list: string): string {
  let best: { p: number; pct: number } | undefined;
  for (const m of list.matchAll(/P(\d+)\s*:\s*([\d.]+)%/g)) {
    const p = +m[1];
    const pct = +m[2];
    if (pct > 0 && (!best || p > best.p)) best = { p, pct };
  }
  return best ? `P${best.p}` : "-";
}
