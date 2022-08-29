# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# zbo@zju.edu.cn
# 2022-08-08
# ============================================================================

import os
import re
import subprocess
import tempfile
import logging

import numpy as np

logger = logging.getLogger(__name__)


def get_moses_multi_bleu(hypotheses, references, lowercase=False):
    """Get the BLEU score using the moses `multi-bleu.perl` script.

    **Script:**
    https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/generic/multi-bleu.perl

    Args:
      hypotheses (list of str): List of predicted values
      references (list of str): List of target values
      lowercase (bool): If true, pass the "-lc" flag to the `multi-bleu.perl` script

    Returns:
      (:class:`np.float32`) The BLEU score as a float32 value.

    Example:

      >>> hypotheses = [
      ...   "The brown fox jumps over the dog 笑",
      ...   "The brown fox jumps over the dog 2 笑"
      ... ]
      >>> references = [
      ...   "The quick brown fox jumps over the lazy dog 笑",
      ...   "The quick brown fox jumps over the lazy dog 笑"
      ... ]
      >>> get_moses_multi_bleu(hypotheses, references, lowercase=True)
      46.51
    """
    if isinstance(hypotheses, list):
        hypotheses = np.array(hypotheses)
    if isinstance(references, list):
        references = np.array(references)

    if np.size(hypotheses) == 0:
        return np.float32(0.0)

    # Get MOSES multi-bleu script
    try:
        multi_bleu_path, _ = six.moves.urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/moses-smt/mosesdecoder/"
            "master/scripts/generic/multi-bleu.perl")
        os.chmod(multi_bleu_path, 0o755)
    except:
        logger.warning("Unable to fetch multi-bleu.perl script")
        return None

    # Dump hypotheses and references to tempfiles
    hypothesis_file = tempfile.NamedTemporaryFile()
    hypothesis_file.write("\n".join(hypotheses).encode("utf-8"))
    hypothesis_file.write(b"\n")
    hypothesis_file.flush()
    reference_file = tempfile.NamedTemporaryFile()
    reference_file.write("\n".join(references).encode("utf-8"))
    reference_file.write(b"\n")
    reference_file.flush()

    # Calculate BLEU using multi-bleu script
    with open(hypothesis_file.name, "r") as read_pred:
        bleu_cmd = [multi_bleu_path]
        if lowercase:
            bleu_cmd += ["-lc"]
        bleu_cmd += [reference_file.name]
        try:
            bleu_out = subprocess.check_output(bleu_cmd, stdin=read_pred, stderr=subprocess.STDOUT)
            bleu_out = bleu_out.decode("utf-8")
            bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
            bleu_score = float(bleu_score)
            bleu_score = np.float32(bleu_score)
        except subprocess.CalledProcessError as error:
            if error.output is not None:
                logger.warning("multi-bleu.perl script returned non-zero exit code")
                logger.warning(error.output)
            bleu_score = None

    # Close temp files
    hypothesis_file.close()
    reference_file.close()

    return bleu_score