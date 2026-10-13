"use client";

import { Lock } from "lucide-react";
import { useEffect, useRef, type KeyboardEvent, type MouseEvent } from "react";

interface PaywallModalProps {
  payAttempted: boolean;
  onPay: () => void;
}

export function PaywallModal({ payAttempted, onPay }: PaywallModalProps) {
  const payRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    payRef.current?.focus();
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previousOverflow;
    };
  }, []);

  const keepFocusWithin = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key !== "Tab") {
      return;
    }
    event.preventDefault();
    payRef.current?.focus();
  };

  const holdFocus = (event: MouseEvent<HTMLDivElement>) => {
    event.preventDefault();
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center px-6"
      onKeyDown={keepFocusWithin}
      onMouseDown={holdFocus}
    >
      <div className="absolute inset-0 bg-canvas/50" aria-hidden="true" />
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="paywall-message"
        className="animate-slide-up relative w-full max-w-[420px] rounded-md border border-hairline-soft bg-canvas px-8 py-9 text-center shadow-modal"
      >
        <div className="mx-auto grid size-12 place-items-center rounded-[30%] bg-canvas-soft">
          <Lock size={20} strokeWidth={1.75} className="text-ink" aria-hidden="true" />
        </div>

        <p
          id="paywall-message"
          className="mx-auto mt-5 max-w-[19rem] text-xl leading-[1.4] font-normal text-ink"
        >
          To unlock your download, please pay $20.00.
        </p>

        <button
          ref={payRef}
          type="button"
          onClick={onPay}
          className="mt-6 inline-flex h-11 w-full cursor-pointer items-center justify-center rounded-full bg-ink px-6 text-base font-semibold text-canvas outline-none transition-transform duration-150 active:scale-[0.98] focus-visible:ring-2 focus-visible:ring-ink focus-visible:ring-offset-4"
        >
          {payAttempted ? "Nice try." : "Pay $20.00"}
        </button>
      </div>
    </div>
  );
}
