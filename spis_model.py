from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field


class SignInput(BaseModel):
    type: Literal["Raw", "Encoded", "Emergent"]
    source: Literal["Human", "Machine", "Environment"]
    symbol_type: Literal["Iconic", "Indexical", "Symbolic"]


class ContextInput(BaseModel):
    scope: Literal["Local", "Cultural", "Systemic"]
    description: Optional[str]


class MemoryTrace(BaseModel):
    structure: Literal["Episodic", "Distributed", "Hierarchical"]
    fidelity: Literal["Lossless", "Compressed", "Abstracted"]
    trace_strength: Literal["Weak", "Medium", "Strong"]


class NoiseProfile(BaseModel):
    type: Literal["Entropic_Signals", "Compression_Failure"]
    entropy_level: Literal["Low", "Medium", "High"]


class ContextFusionStrategy(BaseModel):
    strategy: Literal["Intersection", "Overlap", "Weighting"]
    resolution: Literal["Heuristic", "Probabilistic", "Fallback"]


class InterpretantOutput(BaseModel):
    mode: Literal["Predictive", "Analogical", "Symbolic"]
    confidence_score: float
    signified_concept: str
    interpretive_entropy: Literal["Low", "Medium", "High"]
    meta_status: Literal["Resolved", "Latent", "Ambiguous"]


class SPISLoop(BaseModel):
    sign: SignInput
    context: ContextInput
    memory: MemoryTrace
    noise: Optional[NoiseProfile]
    fusion: ContextFusionStrategy
    output: InterpretantOutput


# Example instantiation
if __name__ == "__main__":
    spis_example = SPISLoop(
        sign=SignInput(type="Emergent", source="Machine", symbol_type="Symbolic"),
        context=ContextInput(scope="Systemic", description="AI discourse in online media"),
        memory=MemoryTrace(structure="Distributed", fidelity="Compressed", trace_strength="Strong"),
        noise=NoiseProfile(type="Entropic_Signals", entropy_level="Medium"),
        fusion=ContextFusionStrategy(strategy="Weighting", resolution="Heuristic"),
        output=InterpretantOutput(
            mode="Predictive",
            confidence_score=0.88,
            signified_concept="Synthetic semiotic memory",
            interpretive_entropy="Low",
            meta_status="Resolved"
        )
    )
    print(spis_example.json(indent=2))
