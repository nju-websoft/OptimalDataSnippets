from dataclasses import dataclass

@dataclass(frozen=True)
class Triple:
    subject: int
    predicate: int
    object: int

    def __lt__(self, other):
        return (self.subject, self.predicate, self.object) < (other.subject, other.predicate, other.object)

    def __eq__(self, other):
        if isinstance(other, Triple):
            return (self.subject, self.predicate, self.object) == (other.subject, other.predicate, other.object)
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.subject, self.predicate, self.object))

    def __repr__(self):
        return f"{self.subject} {self.predicate} {self.object}"

@dataclass(frozen=True)
class WeightedTriple(Triple):
    prpW: float = 0.0
    kwsW: float = 0.0
    sentW: float = 0.0
    sedpW: float = 0.0
    oentW: float = 0.0
    oedpW: float = 0.0

    def __repr__(self):
        return f"{self.subject} {self.predicate} {self.object}"

    @property
    def data_weight(self):
        return self.prpW + self.sentW + self.sedpW + self.oentW + self.oedpW


class ConnectedComponent:
    def __init__(self):
        self.triples_dict = {}
        self.vertex_to_index = {}
        self.index = 0
        self.current_max_size = 0
    
    def add(self, triple: WeightedTriple):
        if triple.subject not in self.vertex_to_index and triple.object not in self.vertex_to_index:
            self.triples_dict[self.index] = set([triple])
            self.vertex_to_index[triple.subject] = self.index
            self.vertex_to_index[triple.object] = self.index
            self.index += 1
        elif triple.subject in self.vertex_to_index and triple.object not in self.vertex_to_index:
            self.triples_dict[self.vertex_to_index[triple.subject]].add(triple)
            self.vertex_to_index[triple.object] = self.vertex_to_index[triple.subject]
        elif triple.object in self.vertex_to_index and triple.subject not in self.vertex_to_index:
            self.triples_dict[self.vertex_to_index[triple.object]].add(triple)
            self.vertex_to_index[triple.subject] = self.vertex_to_index[triple.object]
        else:
            if self.vertex_to_index[triple.subject] != self.vertex_to_index[triple.object]:
                self.triples_dict[self.vertex_to_index[triple.subject]].update(self.triples_dict[self.vertex_to_index[triple.object]])
                to_del_index = self.vertex_to_index[triple.object]
                for transfered_triple in self.triples_dict[to_del_index]:
                    self.vertex_to_index[transfered_triple.subject] = self.vertex_to_index[triple.subject]
                    self.vertex_to_index[transfered_triple.object] = self.vertex_to_index[triple.subject]
                del self.triples_dict[to_del_index]
                
            self.triples_dict[self.vertex_to_index[triple.subject]].add(triple)  
        self.current_max_size = self.max_size

    def cover_gain(self, triple: WeightedTriple):
        if triple.subject not in self.vertex_to_index and triple.object not in self.vertex_to_index:
            return max(1 - self.current_max_size, 0)
        elif triple.subject in self.vertex_to_index and triple.object not in self.vertex_to_index:
            return max(len(self.triples_dict[self.vertex_to_index[triple.subject]]) + 1 - self.current_max_size, 0)
        elif triple.object in self.vertex_to_index and triple.subject not in self.vertex_to_index:
            return max(len(self.triples_dict[self.vertex_to_index[triple.object]]) + 1 - self.current_max_size, 0)
        else:
            if triple in self.triples_dict[self.vertex_to_index[triple.subject]]:
                return 0
            elif self.vertex_to_index[triple.subject] != self.vertex_to_index[triple.object]:
                return max(len(self.triples_dict[self.vertex_to_index[triple.subject]]) + 
                           len(self.triples_dict[self.vertex_to_index[triple.object]]) + 
                           1 - self.current_max_size, 0)
            else:
                return max(len(self.triples_dict[self.vertex_to_index[triple.subject]]) + 1 - self.current_max_size, 0)

    @property
    def max_size(self):
        return max(len(triple_set) for _, triple_set in self.triples_dict.items())
