import re
from unidecode import unidecode


class AbcTokenizer:
    """
    A class to tokenize ABC music notation into token ids for machine learning models, and decode them back.
    """
    def __init__(self):
        self.special_tokens = {"PAD": 0, "MSK": 1, "SOS": 2, "EOS": 3}
        self.delimiter_tokens = ["|:", "::", ":|", "[|", "||", "|]", "|"]
        self.regex_pattern = '(' + '|'.join(map(re.escape, self.delimiter_tokens)) + ')'
        self.vocab_size = 128

    def __call__(self, abc_string, patch_len=20, add_special_patches=False):
        """
        Encodes ABC music notation string into patches.

        Parameters:
            abc_string (str): ABC notation string.
            patch_len (int): The target length for each patch. Default is 20.
            add_special_patches (bool): Whether to add special patches at the beginning and end. Default is False.

        Returns:
            list: A list of patches encoded from the input ABC string.
        """
        abc_lines = [unidecode(line) for line in abc_string.split('\n') if line.strip()]
        tune_data, patches = "", []

        for line in abc_lines:
            if self._is_header_info(line):
                if tune_data:
                    patches.extend(self._process_tune_data(tune_data, patch_len))
                    tune_data = ""
                patches.append(self.bar_to_patch(line + '\n', patch_len))
            else:
                tune_data += line + '\n'

        if tune_data:
            patches.extend(self._process_tune_data(tune_data, patch_len))

        if add_special_patches:
            patches = self._add_special_patches(patches, patch_len)

        return patches

    def decode(self, patches):
        """
        Decodes patches back into ABC music notation string.

        Parameters:
            patches (list): A list of patches to be decoded.

        Returns:
            str: The decoded ABC music notation string.
        """
        return ''.join(self.patch_to_bar(patch) for patch in patches)

    def split_bars(self, tune_data):
        """Splits tune data into individual bars based on delimiter tokens."""
        bars = re.split(self.regex_pattern, ''.join(tune_data))
        bars = [bar for bar in bars if bar]  # Remove empty strings
        return self._merge_bars_with_delimiters(bars)

    def bar_to_patch(self, bar, patch_size):
        """Converts a music bar into a patch of specified length."""
        patch = [self.special_tokens["SOS"]] + [ord(c) for c in bar] + [self.special_tokens["EOS"]]
        return self._finalize_patch(patch, patch_size)

    def patch_to_bar(self, patch):
        """Converts a patch back into a music bar."""
        return ''.join(chr(i) for i in patch if i > self.special_tokens["EOS"])

    def _is_header_info(self, line):
        """Checks if a line from the ABC notation is header information."""
        return len(line) > 1 and (line[0].isalpha() and line[1] == ':' or line.startswith('%%score'))

    def _process_tune_data(self, tune_data, patch_size):
        """Helper method to process tune data into patches."""
        bars = self.split_bars(tune_data)
        return [self.bar_to_patch(bar, patch_size) for bar in bars]

    def _merge_bars_with_delimiters(self, bars):
        """Merges bars with their preceding delimiters, if any."""
        if bars[0] in self.delimiter_tokens:
            bars[1] = bars[0] + bars[1]
            bars.pop(0)
        # Merge every two elements
        merged_list = [bars[i] + bars[i + 1] if i + 1 < len(bars) else bars[i]
                       for i in range(0, len(bars), 2)]

        return merged_list

    def _finalize_patch(self, patch, patch_size):
        """Trims or pads the patch to the final patch size."""
        patch = patch[:patch_size] + [self.special_tokens["PAD"]] * (patch_size - len(patch))
        return patch

    def _add_special_patches(self, patches, patch_size):
        """Adds special starting and ending patches."""
        sos_patch = [self.special_tokens["SOS"]] * (patch_size - 1) + [self.special_tokens["EOS"]]
        eos_patch = [self.special_tokens["SOS"]] + [self.special_tokens["EOS"]] * (patch_size - 1)
        return [sos_patch] + patches + [eos_patch]


# Example usage
if __name__ == "__main__":
    tokenizer = AbcTokenizer()
    abc_string_test = '''X:1
L:1/8
M:6/8
K:Bb
 F | B2 d c2 f | edc B2 F | GAB cec | BAG FGA | B2 d c2 f | edc B2 F | Gec AGA | B2 d B2 |: 
 !fermata!F | D2 F D2 F | EGB cED | C2 E C2 E | DFA Bdf | geg fdb | gab [df]bb | dba gf=e |1 
 fff f2 :|2 fgf _edc!D.C.! ||
    '''  # Your actual ABC notation string
    encoded_patches = tokenizer(abc_string_test)
    decoded_music = tokenizer.decode(encoded_patches)
    print(decoded_music)

    # print("decoded_music:", decoded_music)
