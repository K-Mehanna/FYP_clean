from typing import Optional

class Case:
    def __init__(self, id: str, characterisation, outcome: Optional[int]=None, is_default: bool=False):
        self.id = id
        self.is_default_case = is_default
        self.characterisation = characterisation
        self.is_new_case = outcome is None
        self.outcome = outcome
        
        assert not (self.is_default_case and self.is_new_case)
    
    def different_outcome_to(self, other: 'Case') -> bool:
        return self.outcome != other.outcome
    
    def same_outcome_to(self, other: 'Case') -> bool:
        return self.outcome == other.outcome
    
    def __eq__(self, other: 'Case') -> bool:
        if isinstance(other, Case):
            return self.characterisation == other.characterisation and self.outcome == other.outcome
        
        return False 
    
    def _convert_set_to_str(self, characterisation: set):
        if self.is_default_case and len(characterisation) == 0:
            factors = "\u2205" 
        else:
            sorted_factors = sorted(frozenset(characterisation))
            factors = "{" + ", ".join(sorted_factors) + "}"
            
        return factors
        
    def __str__(self) -> str:
        characterisation = self.characterisation
        if isinstance(self.characterisation, set):
            characterisation = self._convert_set_to_str(characterisation)
        
        outcome = "?" if self.is_new_case else self.outcome
            
        return f"({characterisation}, {outcome})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __hash__(self) -> int:
        return hash((self.id, str(self.characterisation), self.outcome))
        