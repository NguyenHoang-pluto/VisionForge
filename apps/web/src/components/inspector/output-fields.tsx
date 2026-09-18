"use client";

import type { Encoder, PlannerCapabilities, Resolution } from "@/lib/api";
import { useT, type MessageKey } from "@/lib/i18n";
import { useEditorStore } from "@/stores/editor-store";
import { Field, Select } from "@/components/ui";

/**
 * Resolution and encoder, as one pair of controls (Phase 12).
 *
 * They are one decision: 8K exists only on the GPU, so choosing 8K switches
 * the encoder to the GPU, and choosing the CPU steps a GPU-only size down to
 * 4K. Shared by the AI Edit and Export panels so both say the same thing.
 */
export function OutputSizeFields({
  capabilities,
  hint,
  prefix,
}: {
  capabilities: PlannerCapabilities | undefined;
  hint?: string;
  prefix: "ai" | "export";
}) {
  const t = useT();
  const resolution = useEditorStore((s) => s.resolution);
  const encoder = useEditorStore((s) => s.encoder);
  const setOutput = useEditorStore((s) => s.setOutput);

  const resolutions = capabilities?.resolutions ?? ["720p", "1080p"];
  const encoders = capabilities?.encoders ?? ["cpu"];
  const gpuOnly = new Set(capabilities?.gpu_only_resolutions ?? []);
  const gpuAvailable = encoders.includes("gpu");

  return (
    <>
      <Field label={t(`${prefix}.resolution` as MessageKey)} hint={hint}>
        <Select
          value={resolution}
          aria-label={t(`${prefix}.resolution.label` as MessageKey)}
          onChange={(event) => {
            const next = event.target.value as Resolution;
            // 8K is GPU only: picking it picks the encoder that can make it.
            setOutput(gpuOnly.has(next) ? { resolution: next, encoder: "gpu" } : { resolution: next });
          }}
        >
          {resolutions.map((value) => (
            <option key={value} value={value}>
              {t(`resolution.${value}` as MessageKey)}
              {gpuOnly.has(value) ? ` · ${t("encoder.gpuOnly")}` : ""}
            </option>
          ))}
        </Select>
      </Field>

      <Field
        label={t("encoder.label")}
        hint={gpuAvailable ? t(`encoder.${encoder}.hint` as MessageKey) : t("encoder.noGpu")}
      >
        <Select
          value={encoder}
          aria-label={t("encoder.label")}
          disabled={!gpuAvailable}
          onChange={(event) => {
            const next = event.target.value as Encoder;
            // The CPU does not do 8K; step down to the largest size it does.
            setOutput(
              next === "cpu" && gpuOnly.has(resolution)
                ? { encoder: next, resolution: "2160p" }
                : { encoder: next },
            );
          }}
        >
          {encoders.map((value) => (
            <option key={value} value={value}>
              {t(`encoder.${value}` as MessageKey)}
            </option>
          ))}
        </Select>
      </Field>
    </>
  );
}
