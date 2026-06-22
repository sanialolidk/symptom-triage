export type Page = "triage" | "evaluation";

export interface SymptomOption {
  id: string;
  label: string;
}

export interface Meta {
  pathologies: string[];
  symptom_catalog: SymptomOption[];
  eval_notes?: Record<string, string>;
}

export interface DifferentialItem {
  pathology: string;
  probability: number;
}

export interface BranchResult {
  model: string;
  top_prediction: string;
  confidence: number;
  differential: DifferentialItem[];
  abstain: boolean;
  abstain_message: string | null;
  modality_weights?: { text: number; structured: number };
}

export interface TriageResponse {
  input: { age: number; sex: string; symptom_count: number; narrative_preview: string };
  structured: BranchResult;
  text_only: BranchResult;
  multimodal: BranchResult;
  modality_analysis: {
    disagreement_score: number;
    modalities_conflict: boolean;
    interpretation: string;
  };
  explainability: { symptoms: { token: string; description: string }[] };
  safety: { abstain_recommended: boolean; rationale: string };
}

export interface ModelMetrics {
  accuracy: number;
  macro_f1: number;
  top3_accuracy: number;
  mrr?: number;
  ece?: number;
  abstain_rate: number;
  abstain_threshold: number;
  accuracy_when_not_abstaining: number;
}

export interface MetricsPayload {
  eval_notes?: Record<string, string>;
  dataset: {
    name: string;
    train_samples: number;
    val_samples?: number;
    test_samples: number;
    n_classes: number;
    pathologies: string[];
  };
  structured: ModelMetrics;
  text_only: ModelMetrics;
  multimodal: ModelMetrics;
  ablation_noisy_text?: {
    mean_modality_disagreement: number;
    text_top3_under_noise: number;
    structured_top3_stable: number;
  };
  abstention_policy?: Record<string, number>;
  improvement?: Record<string, number>;
}