#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import annotations

import random
import mesa
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from random import randint
import numpy as np
from mesa.space import MultiGrid


# In[2]:


from enum import Enum


class SocialStatus(Enum):
    PUB = 0
    SUBADULT = 1
    VAGRANT = 2
    ADULT = 3


class Gender(Enum):
    FEMALE = 0
    MALE = 1


# In[3]:


# Config
AGE_OF_SETTLEMENT = (20, 84)
AGE_OF_DISPERSAL = (12, 40)

LITTER_SIZE = (4, 6)
MAX_WOLF_AGE = 132
PACK_AREA = 3


# In[4]:


class WolfAgent(Agent):
    def __init__(self,
                 unique_id: int,
                 _model: mesa.Model,
                 age: int = 0,
                 social_status: SocialStatus = SocialStatus.PUB,
                 age_of_dispersal: int = None,
                 age_of_settlement: int = None,
                 gender: Gender = None):
        super().__init__(unique_id, _model)
        self.age = age
        self.social_status = social_status  # pub, subadult, vagrant, adult

        self.survival_prob = None
        self.alive_status = True
        self.pack: "WolfPack | None" = None
        self.age_of_death = None
        self.age_of_first_reproduction = None
        self.color = "#FF0000"
        if gender:
            self.gender = gender
        else:
            self.gender = random.choice([Gender.FEMALE, Gender.MALE])

        if age_of_dispersal:
            self.age_of_dispersal = age_of_dispersal
        else:
            self.age_of_dispersal = randint(*AGE_OF_DISPERSAL)

        if age_of_settlement:
            self.age_of_settlement = age_of_settlement
        else:
            self.age_of_settlement = randint(*AGE_OF_SETTLEMENT)

        self.assign_survival_prob()

    def step(self):
        self.assign_social_status()
        self.assign_survival_prob()

    def assign_survival_prob(self):
        if self.social_status == SocialStatus.PUB:
            if self.age < 6:
                self.survival_prob = 0.65 + random.uniform(-0.1, 0.1)
                self.color = "#0000FF"
            else:
                self.survival_prob = 0.85 + random.uniform(-0.13, 0.13)
        elif self.social_status == SocialStatus.SUBADULT:
            self.survival_prob = 0.74 + random.uniform(-0.13, 0.13)
            self.color = "#FFFF00"
        elif self.social_status == SocialStatus.VAGRANT:
            self.survival_prob = 0.42 + random.uniform(-0.1, 0.1)
            self.color = "#00FF00"
        elif self.social_status == SocialStatus.ADULT:
            self.survival_prob = 0.82 + random.uniform(-0.03, 0.03)
            self.color = "#FF0000"

    def assign_social_status(self):
        if self.social_status == SocialStatus.PUB:
            if self.age == 12:
                self.social_status = SocialStatus.SUBADULT

    def disperse(self):
        if self.age >= 10:
            self.social_status = SocialStatus.VAGRANT

        if self.pack is not None:
            self.pack.remove(self)
            self.pack = None

    def death(self):
        self.alive_status = False
        self.age_of_death = self.age

        if self.pack is not None:
            # Setting new pack center
            if self.pack.alpha_f == self and self.pack.alpha_m is not None:
                self.model.grid.move_agent(self.pack.alpha_m, self.pos)
            self.pack.remove(self)

        self.model.grid.remove_agent(self)
        self.model.schedule.remove(self)

    def __str__(self):
        return f"Wolf ({self.unique_id}): {self.gender}, {self.age}yo, {self.social_status}"


# In[5]:


class WolfPack:
    def __init__(self,
                 _model: WolfModel,
                 pack_id: int):
        self.model = _model
        self.pack_id = pack_id
        self.alpha_m = None
        self.alpha_f = None
        self.wolves: List[WolfAgent] = []
        self.breeding_time_together = 0

    def append(self, wolf):
        self.wolves.append(wolf)

    def remove(self, wolf):
        if wolf != self.alpha_m and wolf != self.alpha_f:
            self.wolves.remove(wolf)
        if self.alpha_f == wolf:
            self.alpha_f = None
        elif self.alpha_m == wolf:
            self.alpha_m = None
        wolf.pack = None

    def _birth_of_wolf(self):
        new_wolf = WolfAgent(self.model.next_id(), self.model)
        self.individual_joins(new_wolf)
        self.model.add_agent(new_wolf)
        self.model.set_on_grid(new_wolf)

    def _should_reproduce(self):
        return 0.79 + random.uniform(-0.05, 0.05) > random.random()

    def reproduction(self):
        if self.alpha_f is not None and self.alpha_m is not None:
            if self.breeding_time_together > 0 or self._should_reproduce():
                for _ in range(randint(*LITTER_SIZE)):
                    self._birth_of_wolf()
                self.breeding_time_together += 1

    def individual_joins(self, wolf: WolfAgent, alpha: bool = False):
        if alpha:
            if wolf.gender == Gender.MALE:
                self.alpha_m = wolf
                self.alpha_m.social_status = SocialStatus.ADULT
            else:
                self.alpha_f = wolf
                self.alpha_f.social_status = SocialStatus.ADULT
            self.breeding_time_together = 0
        else:
            self.wolves.append(wolf)
        wolf.pack = self

    def __str__(self):
        return f"Pack {self.pack_id}: {len(self.wolves)} wolves, {self.alpha_m}, {self.alpha_f}"


# In[6]:


from typing import Any, List


class WolfModel(Model):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(240, 180, False)
        self.next_agent_id = 0
        self.next_pack_id_ = 0
        self.packs: List[WolfPack] = []
        self.step_ = 0
        self.month = 0

        self.initialize_population()

        self.datacollector = DataCollector(model_reporters={
            "wolves_population": get_wolves_amount,
            "packs_population": get_pack_amount,
            "pairs_amount": get_pairs_amount,
            "pubs_amount": get_pubs_amount,
            "subadults_amount": get_subadults_amount,
            "vagrants_amount": get_vagrants_amount,
            "adults_amount": get_adults_amount,
            "average_age": get_avg_age
        })

    def step(self):
        self.step_ += 1
        self.month = self.step_ % 12
        self.datacollector.collect(self)

        self.survival_of_individuals()
        self.dispersal_of_individuals()
        self.remove_packs()
        self.settle_in_packs()
        self.ageing_of_individuals()
        self.transition_of_individuals()

        if self.month == 5:
            self.reproduction_of_individuals()

        self.move_wolves_on_grid()

    def add_agent(self, agent):
        self.schedule.add(agent)

    def next_id(self):
        self.next_agent_id += 1
        return self.next_agent_id

    def next_pack_id(self):
        self.next_pack_id_ += 1
        return self.next_pack_id_

    @property
    def female_vagrants(self) -> list:
        is_female_vagrant = lambda w: (w.social_status == SocialStatus.VAGRANT
                                       and w.gender == Gender.FEMALE
                                       and w.age == w.age_of_dispersal)
        return [wolf for wolf in self.schedule.agents if is_female_vagrant(wolf)]

    @property
    def male_vagrants(self) -> list:
        is_male_vagrant = lambda w: (w.social_status == SocialStatus.VAGRANT
                                     and w.gender == Gender.MALE
                                     and w.age == w.age_of_dispersal)
        return [wolf for wolf in self.schedule.agents if is_male_vagrant(wolf)]

    def find_missing_female_alpha(self, pack: WolfPack):
        if len(self.female_vagrants) == 0:
            return

        pack.individual_joins(self.female_vagrants[0], True)
        self.grid.move_agent(pack.alpha_f, pack.alpha_m.pos)
        self.set_on_grid(pack.alpha_m)

    def find_missing_male_alpha(self, pack: WolfPack):
        if len(self.male_vagrants) == 0:
            return

        pack.individual_joins(self.male_vagrants[0], True)

    def pair_up(self):
        males = self.male_vagrants
        females = self.female_vagrants
        for i in range(min(len(females), len(males))):
            new_pack = WolfPack(self, self.next_pack_id())
            new_pack.individual_joins(males[i], True)
            new_pack.individual_joins(females[i], True)
            self.packs.append(new_pack)
            self.set_on_grid(females[i])
            self.set_on_grid(males[i])

    def vagrant_female_settles(self):
        for wolf in self.female_vagrants:
            new_pack = WolfPack(self, self.next_pack_id())
            new_pack.individual_joins(wolf, True)
            self.packs.append(new_pack)

    def reproduction_of_individuals(self):
        for pack in self.packs:
            pack.reproduction()

    def survival_of_individuals(self):
        for wolf in self.schedule.agents:
            if random.uniform(0, 0.78) > wolf.survival_prob or wolf.age == MAX_WOLF_AGE:
                wolf.death()

    def dispersal_of_individuals(self):
        for pack in self.packs:
            for wolf in pack.wolves:
                if wolf.age == wolf.age_of_dispersal and (
                        wolf.social_status == SocialStatus.PUB or wolf.social_status == SocialStatus.SUBADULT):
                    wolf.disperse()

    def remove_packs(self):
        for pack in self.packs:
            if pack.alpha_m is None and pack.alpha_f is None:
                for wolf in pack.wolves:
                    if wolf.age <= 6:
                        wolf.death()
                    else:
                        wolf.disperse()
                self.packs.remove(pack)

    def settle_in_packs(self):
        for pack in self.packs:
            if pack.alpha_m is None:
                self.find_missing_male_alpha(pack)
            elif pack.alpha_f is None:
                self.find_missing_female_alpha(pack)

        self.pair_up()
        self.vagrant_female_settles()

    def ageing_of_individuals(self):
        for wolf in self.schedule.agents:
            wolf.age += 1

    def transition_of_individuals(self):
        for wolf in self.schedule.agents:
            wolf.step()

    # SETTING UP THE INITIAL POPULATION OF WOLVES AND NEWBORN WOLVES AND NEW PACKS
    def set_on_grid(self, agent):
        if agent.pack is not None and agent.pack.alpha_f == agent:
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            while len(self.grid.get_neighbors((x, y), True, False, PACK_AREA)) != 0:
                x = self.random.randrange(self.grid.width)
                y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))
        elif agent.pack is not None:
            possible_cells = self.grid.get_neighborhood(agent.pack.alpha_f.pos, True, False, PACK_AREA)
            random.shuffle(possible_cells)
            for cell in possible_cells:
                if self.grid.is_cell_empty(cell):
                    self.grid.place_agent(agent, cell)
                    return
        else:
            self.grid.place_agent(agent, (0, 0))
            self.grid.move_to_empty(agent)

    def move_wolves_on_grid(self):
        for wolf in self.schedule.agents:
            if wolf.pack is not None and wolf.pack.alpha_f == wolf:
                pass
            elif wolf.pack is not None and wolf.pack.alpha_f is None and wolf.pack.alpha_m == wolf:
                pass
            elif wolf.pack is not None and wolf.pack.alpha_f is not None:
                possible_cells = self.grid.get_neighborhood(wolf.pack.alpha_f.pos, True, False, PACK_AREA)
                random.shuffle(possible_cells)
                for cell in possible_cells:
                    if self.grid.is_cell_empty(cell):
                        self.grid.move_agent(wolf, cell)
                        break
            elif wolf.pack is not None and wolf.pack.alpha_m is not None:
                possible_cells = self.grid.get_neighborhood(wolf.pack.alpha_m.pos, True, False, PACK_AREA)
                random.shuffle(possible_cells)
                for cell in possible_cells:
                    if self.grid.is_cell_empty(cell):
                        self.grid.move_agent(wolf, cell)
                        break

            else:
                self.grid.move_to_empty(wolf)

    def initialize_population(self):
        for _ in range(8):
            alfa_m = WolfAgent(
                self.next_id(),
                self,
                social_status=SocialStatus.ADULT,
                age=random.randint(30, 80),
                gender=Gender.MALE
            )
            alfa_f = WolfAgent(
                self.next_id(),
                self,
                social_status=SocialStatus.ADULT,
                age=random.randint(30, 80),
                gender=Gender.FEMALE
            )
            new_pack = WolfPack(self, self.next_pack_id())
            new_pack.individual_joins(alfa_f, True)
            new_pack.individual_joins(alfa_m, True)

            self.packs.append(new_pack)
            self.add_agent(alfa_m)
            self.add_agent(alfa_f)
            self.set_on_grid(alfa_f)
            self.set_on_grid(alfa_m)
            for _ in range(4):
                new_wolf = WolfAgent(self.next_id(), self, social_status=SocialStatus.PUB, age=3)
                new_pack.individual_joins(new_wolf)
                self.add_agent(new_wolf)
                self.set_on_grid(new_wolf)

            new_pack.breeding_time_together = 1

        for _ in range(10):
            new_wolf = WolfAgent(self.next_id(), self, social_status=SocialStatus.PUB, age=random.randint(7, 12))
            self.add_agent(new_wolf)
            self.set_on_grid(new_wolf)

        for _ in range(12):
            new_wolf = WolfAgent(self.next_id(), self, social_status=SocialStatus.VAGRANT, age=random.randint(12, 70))
            self.add_agent(new_wolf)
            self.set_on_grid(new_wolf)

        for _ in range(9):
            alfa_m = WolfAgent(
                self.next_id(),
                self,
                social_status=SocialStatus.ADULT,
                age=random.randint(30, 80),
                gender=Gender.MALE
            )
            alfa_f = WolfAgent(
                self.next_id(),
                self,
                social_status=SocialStatus.ADULT,
                age=random.randint(30, 80),
                gender=Gender.FEMALE
            )
            new_pack = WolfPack(self, self.next_pack_id())
            new_pack.individual_joins(alfa_m, True)
            new_pack.individual_joins(alfa_f, True)

            self.packs.append(new_pack)
            self.add_agent(alfa_m)
            self.add_agent(alfa_f)
            self.set_on_grid(alfa_f)
            self.set_on_grid(alfa_m)


# In[7]:


def get_wolves_amount(model):
    return len(model.schedule.agents)


def get_pack_amount(model):
    amount = 0
    for pack in model.packs:
        if len(pack.wolves) != 0:
            amount += 1
    return amount


def get_pairs_amount(model):
    amount = 0
    for pack in model.packs:
        if len(pack.wolves) == 0:
            amount += 1
    return amount


def get_pubs_amount(model):
    amount = 0
    for wolf in model.schedule.agents:
        if wolf.social_status == SocialStatus.PUB:
            amount += 1
    return amount


def get_subadults_amount(model):
    amount = 0
    for wolf in model.schedule.agents:
        if wolf.social_status == SocialStatus.SUBADULT:
            amount += 1
    return amount


def get_vagrants_amount(model):
    amount = 0
    for wolf in model.schedule.agents:
        if wolf.social_status == SocialStatus.VAGRANT:
            amount += 1
    return amount


def get_adults_amount(model):
    amount = 0
    for wolf in model.schedule.agents:
        if wolf.gender == Gender.FEMALE:
            amount += 1
    return amount


def get_avg_age(model):
    ages = [wolf.age for wolf in model.schedule.agents]
    return np.mean(ages)


# In[8]:


model = WolfModel()

months = 145
for _ in range(months):
    model.step()

run_stats = model.datacollector.get_model_vars_dataframe()


# In[9]:


from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer


def agent_portrayal(agent):
    portrayal = {"Shape": "circle",
                 "Filled": "false",
                 "Layer": 2,
                 "Color": agent.color,
                 "r": 0.5}
    return portrayal


measures = ["wolves_population", "packs_population", "pairs_amount", "pubs_amount", "subadults_amount",
            "vagrants_amount", "adults_amount", "average_age"]
wolves_population = ChartModule([{"label": "wolves_population",
                                  "Label": "wolves_population",
                                  "Color": "#0000FF"}],
                                data_collector_name='datacollector')

packs_population = ChartModule([{"label": "packs_population",
                                 "Label": "packs_population",
                                 "Color": "#FF0000"}],
                               data_collector_name='datacollector')

pairs_amount = ChartModule([{"label": "pairs_amount",
                             "Label": "pairs_amount",
                             "Color": "#00FF00"}],
                           data_collector_name='datacollector')

legend = ChartModule([{"label": "pubs",
                       "Label": "pubs_amount",
                       "Color": "#0000FF"},
                      {"label": "subadults",
                       "Label": "subadults_amount",
                       "Color": "#FFFF00"},
                      {"label": "vagrants",
                       "Label": "vagrants_amount",
                       "Color": "#00FF00"},
                      {"label": "adults",
                       "Label": "adults_amount",
                       "Color": "#FF0000"}],
                     data_collector_name='datacollector')

grid = CanvasGrid(agent_portrayal, 240, 180, 1200, 900)

server = ModularServer(
    WolfModel,
    [grid, legend, wolves_population, packs_population, pairs_amount],
    "Wolf Model",
    {}
)

server.port = 8521
server.launch()

