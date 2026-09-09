import type { BatchAxis, BatchMode, PlannedJob, RenderRequest } from "./domain";

export interface BatchDefinition {
  mode: BatchMode | "single";
  axes: BatchAxis[];
  count: number;
}

export interface BatchDraft {
  definition: BatchDefinition;
  inputs: Record<string, string>;
  jobs: Array<{ request: RenderRequest; context: PlannedJob["context"] }>;
}

export interface BatchHandle {
  id: string;
  manifest_path: string;
  job_ids: string[];
}

export function createBatchDraft(jobs: PlannedJob[], definition: BatchDefinition): BatchDraft {
  const inputs: Record<string, string> = {};
  const keys = new Map<string, string>();
  const reference = (value: string): string => {
    const present = keys.get(value);
    if (present) return present;
    const key = `input_${keys.size}`;
    keys.set(value, key);
    inputs[key] = value;
    return key;
  };

  return {
    definition: {
      mode: definition.mode,
      axes: definition.axes.map((axis) => ({ ...axis })),
      count: definition.count,
    },
    inputs,
    jobs: jobs.map((job) => {
      const request: RenderRequest = {
        ...job.request,
        loras: job.request.loras.map((lora) => ({ ...lora })),
        ...(job.request.control ? { control: { ...job.request.control, image: reference(job.request.control.image) } } : {}),
      };
      if (job.request.init_image) request.init_image = reference(job.request.init_image);
      if (job.request.mask) request.mask = reference(job.request.mask);
      return {
        request,
        context: { ...job.context, axes: { ...job.context.axes } },
      };
    }),
  };
}
