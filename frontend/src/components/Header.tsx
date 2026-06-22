interface HeaderProps {
  page: "triage" | "evaluation";
  onPageChange: (page: "triage" | "evaluation") => void;
}

const TABS = [
  { id: "triage" as const, label: "Triage" },
  { id: "evaluation" as const, label: "Results" },
];

export function Header({ page, onPageChange }: HeaderProps) {
  return (
    <header className="site-header">
      <div className="site-header-top">
        <div>
          <h1>Symptom Triage</h1>
          <p>Rank conditions from a symptom checklist and what the patient actually says.</p>
        </div>
        <div className="header-tags">
          <span className="header-tag">DDXPlus</span>
          <span className="header-tag">Structured + text + fusion</span>
        </div>
      </div>
      <nav className="nav-tabs" aria-label="Main">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            type="button"
            className={`nav-tab ${page === tab.id ? "active" : ""}`}
            onClick={() => onPageChange(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </nav>
    </header>
  );
}