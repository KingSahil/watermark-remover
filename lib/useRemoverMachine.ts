"use client";

import { useSyncExternalStore } from "react";
import type { RemoverSnapshot, WatermarkRemoverMachine } from "./machine";

export function useRemoverMachine(
  machine: WatermarkRemoverMachine,
): RemoverSnapshot {
  return useSyncExternalStore(
    machine.subscribe,
    machine.getSnapshot,
    machine.getServerSnapshot,
  );
}
