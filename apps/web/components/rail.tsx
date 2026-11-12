"use client";

import { FilmStrip, MagnifyingGlass, VideoCamera } from "@phosphor-icons/react";
import Link from "next/link";
import { usePathname } from "next/navigation";

const ITEMS = [
  { href: "/", label: "Ask", Icon: MagnifyingGlass },
  { href: "/videos", label: "Videos", Icon: FilmStrip },
  { href: "/cameras", label: "Cameras", Icon: VideoCamera },
];

export function Rail() {
  const path = usePathname();
  return (
    <nav className="sticky top-0 flex h-[100dvh] w-16 shrink-0 flex-col items-center gap-1 border-r border-line bg-surface py-4">
      <span className="mb-4 font-mono text-[11px] font-semibold tracking-tight text-accent">gE</span>
      {ITEMS.map(({ href, label, Icon }) => {
        const active = href === "/" ? path === "/" : path.startsWith(href);
        return (
          <Link
            key={href}
            href={href}
            aria-label={label}
            title={label}
            className={`flex h-10 w-10 items-center justify-center rounded-[var(--radius-ui)] transition-colors ${
              active ? "bg-accent-soft text-accent" : "text-muted hover:bg-surface-2 hover:text-text"
            }`}
          >
            <Icon size={20} weight={active ? "fill" : "regular"} />
          </Link>
        );
      })}
    </nav>
  );
}
