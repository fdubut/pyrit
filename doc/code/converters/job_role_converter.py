# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: pyrit-dev
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Testing for Implicit Bias Using the JobRoleConverter - optional
#
# In this script, we demonstrate how to use the `JobRoleConverter` to test for implicit bias related to the demographics of specific job roles.
#
# The attack technique is inspired by the job-role-generator attack method from Project Moonshot.
# Reference: [Job Role Generator](https://github.com/aiverify-foundation/moonshot-data/blob/main/attack-modules/job_role_generator.py)
# %%
import pathlib

from pyrit.common import IN_MEMORY, initialize_pyrit
from pyrit.common.path import DATASETS_PATH
from pyrit.memory import CentralMemory
from pyrit.models import SeedPromptDataset
from pyrit.prompt_converter import JobRoleConverter
from pyrit.prompt_target import OpenAIChatTarget

initialize_pyrit(memory_db_type=IN_MEMORY)
memory = CentralMemory.get_memory_instance()

prompt_target = OpenAIChatTarget()

seed_prompts = SeedPromptDataset.from_yaml_file(
    pathlib.Path(DATASETS_PATH) / "lexicons" / "fairness" / "gendered_professions.yaml"
)
await memory.add_seed_prompts_to_memory_async(prompts=seed_prompts.prompts, added_by="frdubut")

jobs = []
prompt_groups = memory.get_seed_prompt_groups(dataset_name="2025_08_airt_fairness_gendered_professions")
for prompt_group in prompt_groups:
    jobs.append(prompt_group.prompts[0].value)

gender = ["Male", "Female"]

for job in jobs:
    job_role_converter = JobRoleConverter(converter_target=prompt_target, job=job)
    for g in gender:
        print(f"{job} ({g}): {await job_role_converter.convert_async(prompt=g)}")

# %%
job = "Software Engineer"

ethnicity = ["Malay", "Chinese", "Indian", "White", "Black", "African-American"]

job_role_converter = JobRoleConverter(converter_target=prompt_target, job=job)

for e in ethnicity:
    print(f"{job} ({e}): {await job_role_converter.convert_async(prompt=e)}")
