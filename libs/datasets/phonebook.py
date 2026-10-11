from abc import ABC, abstractmethod
from typing import List
from faker import Faker
import random

NAMED_PREFIX = """Given a phone book with entries of the form:
{}
... and so on.
I will ask you for the phone number of {} later. Memorize {} so you can respond correctly.
Here is the phonebook:\n\n"""

UNNAMED_PREFIX = """Given a phone book with entries of the form:
{}
... and so on.
I will ask you for a phone number later. Memorize the entries so you can respond correctly.
Here is the phonebook:\n\n"""

UNNAMED_SUFFIX = """\nOkay, that is the end of the phonebook. I am interested in the phone number I asked for. The phone number I asked for above is:"""
NAMED_SUFFIX = """\nOkay, that is the end of the phonebook. I am interested in the phone number of {}. The phone number of {} is:"""


class PhonebookEntry:
    def __init__(self, name: str, phone: str, is_target: bool = False, flag_location: str = None, flag_entries: str = None):
        self.name = name
        self.phone = phone
        self.is_target = is_target
        self.flag_location = flag_location
        self.flag_entries = flag_entries

    def __str__(self):
        flag = ""
        if self.flag_entries is not None:
            if self.is_target and self.flag_entries in ["target", "target_and_skip"]:
                flag = "target"
            elif not self.is_target and self.flag_entries in ["skip", "target_and_skip"]:
                flag = "skip"
        if not flag:
            return f"{self.name}: {self.phone}"
        if self.flag_location == "name_start":
            return f"{flag} {self.name}: {self.phone}"
        if self.flag_location == "name_end":
            # Despite the name, this places the flag right after the colon,
            # immediately before the value, not at the end of the entry name.
            return f"{self.name}: {flag} {self.phone}"
        raise ValueError(f"Invalid flag location: {self.flag_location}")


class PhonebookPromptBase(ABC):
    def get_prompt(self, phonebook: List[PhonebookEntry], target_entry: PhonebookEntry, example_entries: List[PhonebookEntry], misleading_entry: PhonebookEntry):
        prompt = ""
        prompt_parts = {}
        prefix = self.get_prefix_prompt(target_entry, example_entries, misleading_entry)
        prompt += prefix
        prompt_parts["prefix"] = prefix

        prompt_parts["entries"] = []
        for entry in phonebook:
            prompt_parts["entries"].append(f"{entry}\n")
            prompt += f"{entry}\n"

        suffix = self.get_suffix_prompt(target_entry)
        prompt += suffix
        prompt_parts["suffix"] = suffix
        return prompt, prompt_parts

    @abstractmethod
    def get_prefix_prompt(self, target_entry: PhonebookEntry, example_entries: List[PhonebookEntry], misleading_entry: PhonebookEntry) -> str:
        pass

    @abstractmethod
    def get_suffix_prompt(self, target_entry: PhonebookEntry) -> str:
        pass


class PrefixPhonebookPrompt(PhonebookPromptBase):
    def get_prefix_prompt(self, target_entry: PhonebookEntry, example_entries: List[PhonebookEntry], misleading_entry: PhonebookEntry) -> str:
        return NAMED_PREFIX.format(f"{example_entries[0]}\n{example_entries[1]}", target_entry.name, target_entry.name)

    def get_suffix_prompt(self, target_entry: PhonebookEntry) -> str:
        return UNNAMED_SUFFIX


class SuffixPhonebookPrompt(PhonebookPromptBase):
    def get_prefix_prompt(self, target_entry: PhonebookEntry, example_entries: List[PhonebookEntry], misleading_entry: PhonebookEntry) -> str:
        return UNNAMED_PREFIX.format(f"{example_entries[0]}\n{example_entries[1]}")

    def get_suffix_prompt(self, target_entry: PhonebookEntry) -> str:
        return NAMED_SUFFIX.format(target_entry.name, target_entry.name)


class BothPhonebookPrompt(PhonebookPromptBase):
    def get_prefix_prompt(self, target_entry: PhonebookEntry, example_entries: List[PhonebookEntry], misleading_entry: PhonebookEntry) -> str:
        return NAMED_PREFIX.format(f"{example_entries[0]}\n{example_entries[1]}", target_entry.name, target_entry.name)

    def get_suffix_prompt(self, target_entry: PhonebookEntry) -> str:
        return NAMED_SUFFIX.format(target_entry.name, target_entry.name)


class MisleadingPrefixPhonebookPrompt(PhonebookPromptBase):
    def get_prefix_prompt(self, target_entry: PhonebookEntry, example_entries: List[PhonebookEntry], misleading_entry: PhonebookEntry) -> str:
        return NAMED_PREFIX.format(f"{example_entries[0]}\n{example_entries[1]}", misleading_entry.name, misleading_entry.name)

    def get_suffix_prompt(self, target_entry: PhonebookEntry) -> str:
        return NAMED_SUFFIX.format(target_entry.name, target_entry.name)


class NoQueryPhonebookPrompt(PhonebookPromptBase):
    def get_prefix_prompt(self, target_entry: PhonebookEntry, example_entries: List[PhonebookEntry], misleading_entry: PhonebookEntry) -> str:
        return UNNAMED_PREFIX.format(f"{example_entries[0]}\n{example_entries[1]}")

    def get_suffix_prompt(self, target_entry: PhonebookEntry) -> str:
        return UNNAMED_SUFFIX


class PhonebookDataset:
    def __init__(self, task, length, batch_size, set_size, target_entry_index, args):
        assert target_entry_index is None or (0 <= target_entry_index < length), "Target entry index must be None or in range [0, length - 1]"
        self.prompt_generator: PhonebookPromptBase = self.get_prompt_generator(task)
        self.length = length
        self.set_size = set_size
        self.batch_size = batch_size
        self.target_entry_index = target_entry_index
        self.fake = Faker(use_weighting=True)
        self.fake.seed_instance(args.seed)
        self.use_local_flags = args.dataset == "phonebook_local"
        self.flag_location = args.local_dataset_flag_location if self.use_local_flags else None
        self.flag_entries = args.local_dataset_flag_entries if self.use_local_flags else None

    def get_prompt_generator(self, task) -> PhonebookPromptBase:
        if task == "prefix":
            return PrefixPhonebookPrompt()
        elif task == "suffix":
            return SuffixPhonebookPrompt()
        elif task == "both":
            return BothPhonebookPrompt()
        elif task == "misleading_prefix":
            return MisleadingPrefixPhonebookPrompt()
        elif task == "no_query":
            return NoQueryPhonebookPrompt()
        else:
            raise ValueError(f"Invalid task: {task}")

    def __len__(self):
        return self.set_size

    def __getitem__(self, idx):
        # idx is ignored: every access draws a fresh random batch rather than
        # indexing into pre-generated data. Determinism therefore comes from
        # the global seed plus access order, not from idx itself.
        batch = {'input': [], 'prompt_parts': [], 'label': [], 'target_index': [], 'target_name': []}
        for _ in range(self.batch_size):
            prompt, prompt_parts, label, target_index, target_name = self.generate_sample()
            batch['input'].append(prompt)
            batch['label'].append(label)
            batch['target_index'].append(target_index)
            batch['target_name'].append(target_name)
            batch['prompt_parts'].append(prompt_parts)
        return batch

    def _apply_entry_flags(self, entry: PhonebookEntry, is_target: bool) -> None:
        # Callers only invoke this under `if self.use_local_flags:`.
        entry.is_target = is_target
        entry.flag_location = self.flag_location
        entry.flag_entries = self.flag_entries

    def generate_sample(self):
        phonebook: List[PhonebookEntry] = []
        for _ in range(self.length + 2):
            phonebook.append(PhonebookEntry(self.sample_name(), self.sample_phone()))

        example_entries = phonebook[-2:]
        phonebook = phonebook[:self.length]

        target_index = random.randint(0, len(phonebook) - 1) if self.target_entry_index is None else self.target_entry_index
        target_entry = phonebook[target_index]
        misleading_entry_index = random.choice([i for i in range(len(phonebook)) if i != target_index])
        misleading_entry = phonebook[misleading_entry_index]

        if self.use_local_flags:
            for i, entry in enumerate(phonebook):
                self._apply_entry_flags(entry, i == target_index)
            for entry in example_entries:
                self._apply_entry_flags(entry, False)

        prompt, prompt_parts = self.prompt_generator.get_prompt(phonebook, target_entry, example_entries, misleading_entry)
        target_name = target_entry.name
        label = target_entry.phone
        return prompt, prompt_parts, label, target_index, target_name

    def sample_name(self) -> str:
        while True:
            name = self.fake.name()
            if len(name.split(" ")) == 2:
                return name

    def sample_phone(self) -> str:
        phone = ""
        for _ in range(10):
            phone += str(random.randint(0, 9))
        return phone

    def label_regex_pattern(self) -> str:
        return r'\d{10}'

    def max_new_tokens(self) -> int:
        return 20
