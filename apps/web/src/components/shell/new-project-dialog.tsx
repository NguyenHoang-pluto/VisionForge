"use client";

import { useT } from "@/lib/i18n";
import { Button, Dialog, Field, TextInput } from "@/components/ui";

/**
 * Creating a project.
 *
 * The only thing a project needs to exist is a name, and asking for more up
 * front -- a description, a format, a target length -- would be asking for
 * decisions that are better made once there is footage to make them about.
 *
 * Lived in `top-bar.tsx` until the navigation rail replaced that component.
 * It is called from two places now (Home and the project switcher), which is
 * the other reason it is its own file.
 */
export function NewProjectDialog({
  open,
  title,
  onTitle,
  onClose,
  onCreate,
}: {
  open: boolean;
  title: string;
  onTitle: (title: string) => void;
  onClose: () => void;
  onCreate: (title: string) => void;
}) {
  const t = useT();

  return (
    <Dialog
      open={open}
      onClose={onClose}
      title={t("top.project.new")}
      description={t("home.empty.body")}
    >
      <form
        className="flex flex-col gap-panel-gap"
        onSubmit={(event) => {
          event.preventDefault();
          if (title.trim()) onCreate(title.trim());
        }}
      >
        <Field label={t("top.project.newTitle")}>
          <TextInput
            autoFocus
            value={title}
            maxLength={200}
            placeholder={t("top.project.newPlaceholder")}
            onChange={(event) => onTitle(event.target.value)}
          />
        </Field>

        <div className="flex justify-end gap-2">
          <Button onClick={onClose}>{t("common.cancel")}</Button>
          <Button type="submit" tone="primary" disabled={!title.trim()}>
            {t("top.project.create")}
          </Button>
        </div>
      </form>
    </Dialog>
  );
}
