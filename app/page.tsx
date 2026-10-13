"use client";

import { useEffect, useState } from "react";
import { PaywallModal } from "@/components/PaywallModal";
import { ProcessingView } from "@/components/ProcessingView";
import { ResultView } from "@/components/ResultView";
import { UploadView } from "@/components/UploadView";
import { WatermarkRemoverMachine } from "@/lib/machine";
import { processImage } from "@/lib/transport";
import { useRemoverMachine } from "@/lib/useRemoverMachine";

export default function Page() {
  const [machine] = useState(() => new WatermarkRemoverMachine(processImage));
  const snapshot = useRemoverMachine(machine);

  useEffect(() => () => machine.destroy(), [machine]);

  switch (snapshot.step) {
    case "processing":
      return (
        <ProcessingView
          progress={snapshot.progress}
          stage={snapshot.stage}
          previewUrl={snapshot.previewUrl}
          finalUrl={snapshot.finalUrl}
          maskUrl={snapshot.maskUrl}
        />
      );

    case "result":
    case "paywall":
      return (
        <>
          <ResultView
            previewUrl={snapshot.previewUrl}
            onDownload={() => machine.requestDownload()}
          />
          {snapshot.step === "paywall" ? (
            <PaywallModal
              payAttempted={snapshot.payAttempted}
              onPay={() => machine.acknowledgePaywall()}
            />
          ) : null}
        </>
      );

    case "upload":
    default:
      return (
        <UploadView
          error={snapshot.error}
          onStart={(file) => machine.attach(file)}
        />
      );
  }
}
