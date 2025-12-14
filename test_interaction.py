import pytest
import numpy as np
from interaction import Interaction

@pytest.fixture
def interaction_rules():
    """Erstellt eine saubere Interaction-Instanz mit definierten Werten."""
    NUM_TYPES = 4
    MAX_DIST = 50.0
    return Interaction(num_types=NUM_TYPES, max_distance=MAX_DIST, friction=0.1)

def test_interaction_initialization(interaction_rules):
    """Prüft, ob die Klasse korrekt initialisiert wurde und Parameter stimmen."""
    assert interaction_rules.num_types == 4
    assert interaction_rules.max_distance == 50.0
    assert interaction_rules.friction == 0.1
    assert interaction_rules.matrix.shape == (4, 4)
    

def test_matrix_diagonal_is_attraction(interaction_rules):
    """Prüft die Regel: Gleiche Typen (Diagonale) müssen Anziehung (+1.0) sein."""
    assert np.all(np.diag(interaction_rules.matrix) == 1.0)

def test_matrix_off_diagonal_is_repulsion(interaction_rules):
    """Prüft die Regel: Ungleiche Typen (Nicht-Diagonale) müssen Abstoßung (-1.0) sein."""
    
    # Holen eines nicht-diagonalen Elements, z.B. Typ 0 auf Typ 2
    assert interaction_rules.matrix[0, 2] == -1.0
    

def test_force_magnitude_at_zero_distance(interaction_rules):
    """Prüft, ob die Kraft bei Abstand 0 maximal ist."""
    # Erwartung: 1.0 * (1.0 - 0/50.0) = 1.0
    force = interaction_rules.calculate_force_magnitude(1.0, 0.0)
    assert force == 1.0

def test_force_magnitude_at_half_distance(interaction_rules):
    """Prüft, ob die Kraft bei halbem Abstand (25.0) halbiert wird."""
    # Erwartung: 1.0 * (1.0 - 25.0/50.0) = 0.5
    force = interaction_rules.calculate_force_magnitude(1.0, 25.0)
    assert np.isclose(force, 0.5)

def test_force_magnitude_beyond_max_distance(interaction_rules):
    """Prüft, ob die Kraft bei Überschreitung der Reichweite Null ist."""
    # Erwartung: Bei 50.1 muss 0.0 zurückgegeben werden.
    force = interaction_rules.calculate_force_magnitude(-1.0, 50.1)
    assert force == 0.0

