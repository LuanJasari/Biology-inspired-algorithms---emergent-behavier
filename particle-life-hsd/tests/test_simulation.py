import numpy as np
import pytest
from simulation import Simulation

# --------- vereinfachte Mock-Klassen ----------
""" damit lässt sich die physikalische Berechnungslogik
der Simulation-Klasse isoliert testen."""

class SimpleparticleMock:
  # Nur die absolut notwendigen Attribute und Methoden
  def __init__(self, pos, types):
    self.positions = pos
    self.types = types
    self.velocities = np.zeros(pos.shape)
    self.accelerations = np.zeros(pos.shape)


class SimpleInteractionMock:
  # Gibt eine feste Regel-Matrix zuruck, ohne Indexierung zu simulieren.
  def __init__(self,rule_value):
    self.rule_value = rule_value


def get_rule_grid(self, types_array):
  # angenommen, es sind 2 Partikel. gibt eine einfache Matrix zuruck.
  # Diagonale ist oft anders, aber wir simulieren nur die Interaktion (Off-Diagonale)
  return np.array([[1.0], self.rule_value, 1.0])

# ----------- Direkter Fixture / Setup  ----------

@pytest.fixture
def basic_simulation():
  # SETUP: Zwei Partikel, leicht voneinander entfernt (abstand 0.5)
  # Regeln: Anziehung (+ 0.5)

  dt = 0.1
  max_r = 1.0
  fiction = 0.5

  positions = np.array([[0.0, 0.0][0.5, 0.0])
  types = np.array([0.0])

  # Vereinfachte Regel: Gleiche Typen ziehen sich an (Force = +0.5)
  mock_particles = SimpleparticleMock(positions, types)
  mock_interaction = SimpleInteractionMock(rule_value=1.0) # Der Wert 1.0 wird multipliziert


  return Simulation(dt, max_r, friction, mock_particles, mock_interaction)

# ------------ Haupt-Test-Szenario ----------------

def test_full_simulation_step(basic_simulation):
  # Testet die gesamte Kette: krafte berechnen und Positionen aktualiesieren.

  sim = basic_simulation

  # 1. Distanz-Prufung: Muss 0.5 sein
  distances = sim.compute_distances()
  assert np.allclose(distances[0,1], 0.5)

  # 2. Gesamt-Kraft-Prufung: muss anziehung sein
  # Krafte = Regel (1.0) * (1 - Distnaz/max_R) = 1.0 * (1 - 0.5/1.0) = 0.5
  total_forces = sim.compute_total_forces()
  # Partikel 0 muss nach links gezogen werden (+X)
  assert np.allclose(total_forces[0],[0.5, 0.0])
  # Partikel 1 muss nach links gezogen werden (-X)
  assert np.allclose(total_forces[1], [-0.5, 0.0])

  # 3. KINEMATIK-Prufung: Ein Schritt (Accelration, Velocity, Position)
  sim.update_accelerations()
  sim.update_velocities()
  sim.update_positions()

  # 3. Kinematik-Prufung: Ein Schritt (Accelration, Velocity, Position)
  sim.update_accelerations()
  sim.update_velocities()
  sim.update_positions()

  # Pruft, ob sich die Partikel aufeinander zu bewegen
  assert sim.particles.positions[0, 0] > 0.0
  assert sim.particles.positions[1, 0] < 0.5

  # Pruft, ob Reibung angewendet wurde (Geschw. ist kleiner als reine a*dt)
  pure_velocity = 0.5 * sim.dt # a * dt = 0.05
  assert sim.particles.velocities[0, 0] < pure_velocity


# -----Edge Case Test (Getrennt) ---------

def test_no_interaction_outside_max_r():
  # testet den Randfall: Distanz ist größer als der max_r

  max_r_small = 0.5
  # Partikel sind 0.1 entfernt
  positions = np.array([[0.0, 0.0], [1.0, 0.0]])
  types = np.array([0, 0])

  mock_particles = SimpleParticleMock(positions, types)
  mock_interaction = SimpleInteractionMock(rule_value=1.0)

  sim = Simulation(dt=0.1, max_r=max_r_small, friction=0.0, particles=mock_particles, interactions=mock_interaction)

  # Die Kraft muss 0 sein, da die Distanz (1.0) > max_r (0.5) ist
  total_forces = sim.compute_total_forces()
  assert np.allclose(total_forces, np.zeros((2, 2)))
