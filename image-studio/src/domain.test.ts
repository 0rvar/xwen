import { describe, expect, test } from 'bun:test';
import { DEFAULT_SETTINGS, buildRequest, parseNumericValues, planJobs, restoreSettings, snapSize, validateRequest, type StudioSettings } from './domain';

const image = { name: 'source.png', data_url: 'data:image/png;base64,source', width: 512, height: 512 };
function settings(overrides: Partial<StudioSettings> = {}): StudioSettings {
  return { ...DEFAULT_SETTINGS, prompt: 'a red bicycle', seed: '47', loras: [], ...overrides };
}

describe('native request modes', () => {
  test('imported dimensions snap to the nearest valid API grid', () => {
    expect(snapSize(1920, 1080)).toEqual({ width: 1920, height: 1088 });
    expect(snapSize(500, 500)).toEqual({ width: 496, height: 512 });
    expect(snapSize(1024, 768)).toEqual({ width: 1024, height: 768 });
    expect(snapSize(1, 1)).toEqual({ width: 64, height: 128 });
    expect(() => snapSize(0, 500)).toThrow();
  });
  test('text mode omits inactive edit and control fields', () => {
    const request = buildRequest(settings({ initImage: image, mask: image, controlImage: image }));
    expect(request).toEqual({ prompt: 'a red bicycle', width: 1024, height: 1024, seed: 47, steps: 8, n: 1, loras: [] });
  });
  test('img2img, masked editing and control can combine with multiple LoRAs', () => {
    const request = buildRequest(settings({ mode: 'inpaint', initImage: image, mask: image, strength: 1, maskBlur: 8,
      controlEnabled: true, controlImage: image, loras: [{ name: '/srv/a.safetensors', weight: -0.5 }, { name: '/srv/b.safetensors', weight: 1.2 }] }));
    expect(request.init_image).toBe(image.data_url);
    expect(request.mask).toBe(image.data_url);
    expect(request.strength).toBe(1);
    expect(request.mask_blur).toBe(8);
    expect(request.control).toEqual({ image: image.data_url, preprocess: 'canny', scale: 0.75, start: 0, end: 0.8 });
    expect(request.loras.map(lora => lora.weight)).toEqual([-0.5, 1.2]);
  });
  test('incomplete inputs fail before submission', () => {
    expect(() => buildRequest(settings({ mode: 'img2img' }))).toThrow('source image');
    expect(() => buildRequest(settings({ mode: 'inpaint', initImage: image }))).toThrow('mask');
    expect(() => buildRequest(settings({ controlEnabled: true }))).toThrow('control image');
  });
  test('size and numeric constraints match image admission', () => {
    for (const override of [{ width: 513 }, { width: 16, height: 16 }, { height: 8208 }, { steps: 51 }, { steps: NaN }, { seed: '9007199254740992' }]) {
      expect(() => buildRequest(settings(override))).toThrow();
    }
    expect(buildRequest(settings({ width: 1536, height: 1024 })).width).toBe(1536);
    const request = buildRequest(settings());
    expect(() => validateRequest({ ...request, strength: 0.5 })).toThrow('source');
    expect(() => validateRequest({ ...request, mask_blur: 4 })).toThrow('mask');
  });
});

describe('batch ranges and matrices', () => {
  test('decimal and descending ranges include the endpoint when reached', () => {
    expect(parseNumericValues('0.4:0.8:0.1')).toEqual([0.4, 0.5, 0.6, 0.7, 0.8]);
    expect(parseNumericValues('3:1:-1')).toEqual([3, 2, 1]);
    expect(parseNumericValues('1:2:0.4')).toEqual([1, 1.4, 1.8]);
    expect(parseNumericValues('0:0:1')).toEqual([0]);
    expect(parseNumericValues('1e2, -0.25, .5')).toEqual([100, -0.25, 0.5]);
    expect(parseNumericValues('9007199254740990:9007199254740991:1')).toEqual([9007199254740990, 9007199254740991]);
  });
  test('invalid and unbounded ranges are refused', () => {
    for (const value of ['', '1,,2', '1:3', '1:3:0', '3:1:1', '0:1000:1', 'NaN', 'Infinity', '0x20', '1:3:1,5', '1+2']) {
      expect(() => parseNumericValues(value)).toThrow();
    }
    expect(() => parseNumericValues('9007199254740990:9007199254740991:0.1')).toThrow('too small');
  });
  test('matrix crosses every axis and keeps seeds matched across comparisons', () => {
    const jobs = planJobs(settings({ count: 2 }), [
      { parameter: 'steps', values: '4,8' }, { parameter: 'prompt', values: '["red bicycle", "blue bicycle"]' },
    ]);
    expect(jobs).toHaveLength(8);
    expect(jobs.map(job => [job.request.steps, job.request.prompt, job.request.seed])).toEqual([
      [4, 'red bicycle', 47], [4, 'red bicycle', 48], [4, 'blue bicycle', 47], [4, 'blue bicycle', 48],
      [8, 'red bicycle', 47], [8, 'red bicycle', 48], [8, 'blue bicycle', 47], [8, 'blue bicycle', 48],
    ]);
    expect(jobs[7]!.context).toEqual({ mode: 'text', batch_index: 3, axes: { steps: 8, prompt: 'blue bicycle' }, repeat_index: 1 });
    expect(new Set(jobs.map(job => job.id)).size).toBe(8);
  });
  test('paired mode broadcasts singleton axes and rejects unequal non-singletons', () => {
    const jobs = planJobs(settings(), [{ parameter: 'seed', values: '10:12:1' }, { parameter: 'steps', values: '4,6,8' }, { parameter: 'prompt', values: 'one prompt' }], 'paired');
    expect(jobs.map(job => [job.request.seed, job.request.steps])).toEqual([[10, 4], [11, 6], [12, 8]]);
    expect(() => planJobs(settings(), [{ parameter: 'seed', values: '1,2' }, { parameter: 'steps', values: '4,6,8' }], 'paired')).toThrow('same number');
  });
  test('plan snapshots nested LoRAs and control independently', () => {
    const original = settings({ controlEnabled: true, controlImage: image, loras: [{ name: 'style', weight: 1 }] });
    const jobs = planJobs(original, [{ parameter: 'control.scale', values: '0.5,1' }, { parameter: 'loras.0.weight', values: '0,1' }]);
    original.loras[0]!.weight = 100;
    expect(jobs.map(job => job.request.loras[0]!.weight)).toEqual([0, 1, 0, 1]);
    jobs[0]!.request.control!.scale = 0;
    jobs[0]!.request.loras[0]!.weight = -100;
    expect(jobs[1]!.request.control!.scale).toBe(0.5);
    expect(jobs[1]!.request.loras[0]!.weight).toBe(1);
  });
  test('LoRA axes try each adapter with the current weight', () => {
    const jobs = planJobs(settings({ loras: [{ name: '/srv/base.safetensors', weight: 0.7 }, { name: '/srv/other.safetensors', weight: 0.3 }] }), [
      { parameter: 'lora', values: '/srv/one.safetensors\n/srv/two.safetensors' },
    ]);
    expect(jobs.map(job => job.request.loras)).toEqual([
      [{ name: '/srv/one.safetensors', weight: 0.7 }],
      [{ name: '/srv/two.safetensors', weight: 0.7 }],
    ]);
  });
  test('LoRA axes default to a restrained exploratory weight', () => {
    const jobs = planJobs(settings(), [{ parameter: 'lora', values: '/srv/one.safetensors' }]);
    expect(jobs[0]!.request.loras).toEqual([{ name: '/srv/one.safetensors', weight: 0.8 }]);
  });
  test('random seed is resolved once and then explicit in every request', () => {
    const jobs = planJobs(settings({ seed: '', count: 2 }), [{ parameter: 'steps', values: '4,8' }]);
    expect(Number.isSafeInteger(jobs[0]!.request.seed)).toBe(true);
    expect(jobs[1]!.request.seed).toBe(jobs[0]!.request.seed + 1);
    expect(jobs[2]!.request.seed).toBe(jobs[0]!.request.seed);
  });
  test('bad combinations prevent the entire plan, including seed overflow', () => {
    expect(() => planJobs(settings(), [{ parameter: 'steps', values: '4,51' }])).toThrow('Combination 2');
    expect(() => planJobs(settings({ seed: '9007199254740991', count: 2 }))).toThrow('Seed');
    expect(() => planJobs(settings(), [{ parameter: 'strength', values: '0.5' }])).toThrow('requires img2img');
    expect(() => planJobs(settings(), [{ parameter: 'loras.0.weight', values: '0.5' }])).toThrow('Add LoRA');
    expect(() => planJobs(settings(), [{ parameter: 'guidance', values: '7' }])).toThrow('Unknown');
    expect(() => planJobs(settings(), [{ parameter: 'seed', values: '1' }, { parameter: 'seed', values: '2' }])).toThrow('only once');
    expect(() => planJobs(settings({ count: 100 }), [{ parameter: 'seed', values: '1:11:1' }])).toThrow('at most 1000');
  });
  test('a prompt axis can supply an otherwise empty prompt', () => {
    expect(planJobs(settings({ prompt: '' }), [{ parameter: 'prompt', values: 'first\nsecond' }]).map(job => job.request.prompt)).toEqual(['first', 'second']);
    expect(() => planJobs(settings(), [{ parameter: 'prompt', values: '[1,2]' }])).toThrow('nonempty prompts');
  });
  test('control windows are checked after applying all axes', () => {
    const original = settings({ controlEnabled: true, controlImage: image });
    expect(planJobs(original, [{ parameter: 'control.start', values: '0.9' }, { parameter: 'control.end', values: '1' }])[0]!.request.control!.start).toBe(0.9);
    expect(() => planJobs(original, [{ parameter: 'control.start', values: '0.9' }])).toThrow('before control end');
  });
});

test('metadata restores request fields and clears stale source images', () => {
  const request = buildRequest(settings({ mode: 'inpaint', initImage: image, mask: image, strength: 1, controlEnabled: true, controlImage: image }));
  request.init_image = 'inputs/source.png'; request.mask = 'inputs/mask.png';
  const restored = restoreSettings({ request }, settings({ initImage: image }));
  expect(restored.mode).toBe('inpaint'); expect(restored.seed).toBe('47');
  expect(restored.strength).toBe(1); expect(restored.controlEnabled).toBe(true);
  expect(restored.initImage).toBeNull(); expect(restored.mask).toBeNull();
});
