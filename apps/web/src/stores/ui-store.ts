import { create } from "zustand";

/**
 * Ephemeral client-only UI state.
 *
 * Phase 0 rule: server state lives in TanStack Query and is never duplicated
 * here. In Phase 1 the only such state is whether health details are expanded --
 * the store exists to establish the seam the timeline editor will use.
 */
interface UiState {
  showComponentDetail: boolean;
  toggleComponentDetail: () => void;
}

export const useUiStore = create<UiState>((set) => ({
  showComponentDetail: true,
  toggleComponentDetail: () =>
    set((state) => ({ showComponentDetail: !state.showComponentDetail })),
}));
