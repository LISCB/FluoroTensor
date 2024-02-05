# Module for calculating aSNR (arbitrary signal to noise ratio) of a fluorescence trace when the step boundaries are known

# Copyright (C) 2023: Max F. K. Wills & Ian C. Eperon

# FluoroTensor: identification and tracking of colocalised molecules and their stoichiometries in multi-colour single molecule imaging via deep learning
# Max F.K. Wills, Carlos Bueno Alejo, Nikolas Hundt, Andrew J. Hudson, Ian C. Eperon
# bioRxiv 2023.11.21.567874; doi: https://doi.org/10.1101/2023.11.21.567874

# LICENCE:
# This program is made available under the Creative Commons NonCommercial 4.0 licence.


#      Disclaimer of Warranties and Limitation of Liability.
#
#   a. UNLESS OTHERWISE SEPARATELY UNDERTAKEN BY THE LICENSOR, TO THE
#      EXTENT POSSIBLE, THE LICENSOR OFFERS THE LICENSED MATERIAL AS-IS
#      AND AS-AVAILABLE, AND MAKES NO REPRESENTATIONS OR WARRANTIES OF
#      ANY KIND CONCERNING THE LICENSED MATERIAL, WHETHER EXPRESS,
#      IMPLIED, STATUTORY, OR OTHER. THIS INCLUDES, WITHOUT LIMITATION,
#      WARRANTIES OF TITLE, MERCHANTABILITY, FITNESS FOR A PARTICULAR
#      PURPOSE, NON-INFRINGEMENT, ABSENCE OF LATENT OR OTHER DEFECTS,
#      ACCURACY, OR THE PRESENCE OR ABSENCE OF ERRORS, WHETHER OR NOT
#      KNOWN OR DISCOVERABLE. WHERE DISCLAIMERS OF WARRANTIES ARE NOT
#      ALLOWED IN FULL OR IN PART, THIS DISCLAIMER MAY NOT APPLY TO YOU.
#
#   b. TO THE EXTENT POSSIBLE, IN NO EVENT WILL THE LICENSOR BE LIABLE
#      TO YOU ON ANY LEGAL THEORY (INCLUDING, WITHOUT LIMITATION,
#      NEGLIGENCE) OR OTHERWISE FOR ANY DIRECT, SPECIAL, INDIRECT,
#      INCIDENTAL, CONSEQUENTIAL, PUNITIVE, EXEMPLARY, OR OTHER LOSSES,
#      COSTS, EXPENSES, OR DAMAGES ARISING OUT OF THIS PUBLIC LICENSE OR
#      USE OF THE LICENSED MATERIAL, EVEN IF THE LICENSOR HAS BEEN
#      ADVISED OF THE POSSIBILITY OF SUCH LOSSES, COSTS, EXPENSES, OR
#      DAMAGES. WHERE A LIMITATION OF LIABILITY IS NOT ALLOWED IN FULL OR
#      IN PART, THIS LIMITATION MAY NOT APPLY TO YOU.
#
#   c. The disclaimer of warranties and limitation of liability provided
#      above shall be interpreted in a manner that, to the extent
#      possible, most closely approximates an absolute disclaimer and
#      waiver of all liability.

# this function uses the aSNR calculation method in the Xu et al 2019 paper:
# Xu J, Qin G, Luo F, Wang L, Zhao R, Li N, Yuan J, Fang X. Automated Stoichiometry Analysis of Single-Molecule
# #Fluorescence Imaging Traces via Deep Learning. J Am Chem Soc. 2019 May 1;141(17):6976-6985.
# doi: 10.1021/jacs.9b00688. Epub 2019 Apr 18. PMID: 30950273.


import numpy as np


def calculate_asnr(trace, trace_info, min_length):
    """
    This function takes a trace (list) and the trace_info (in FluoroTensor 6.0+ format) and returns
    the aSNR of that trace or False if the trace is invalid, i.e. if it has 5+ steps, 0 steps, P.B. or
    plateaus shorter than 20 frames.
    """

    # Check if the trace has a valid step count and if not, end the function and return boolean: false
    allowed_steps = [1, 2, 3, 4]
    if trace_info[1] not in allowed_steps:
        return False

    # Check if all plateaus are long enough for stats like standard deviation etc. that will be required for aSNR
    step_positions = trace_info[4]
    if len(step_positions) == 0:
        return False
    if step_positions[0] < min_length or trace_info[2] - step_positions[-1] < min_length:
        return False
    failed = False
    for check in range(len(step_positions) - 1):
        if step_positions[check + 1] - step_positions[check] < min_length:
            failed = True
    if failed:
        return False

    # Now that we know the trace is valid we can start calculating parameters needed for aSNR

    # First we must insert the start and end of the trace into the step positions so we can delineate plateaus
    plateau_pos = [0]
    for add in range(len(step_positions)):
        plateau_pos.append(step_positions[add])
    plateau_pos.append(trace_info[2] - 1)
    print(f"         Plateau boundaries: {plateau_pos}")

    # Now we can calculate means / standard deviations of all plateaus
    means = []
    stdevs = []
    for index in range(len(plateau_pos) - 1):
        means.append(np.mean(trace[plateau_pos[index]+1:plateau_pos[index + 1]]))
        stdevs.append(np.std(trace[plateau_pos[index]+1:plateau_pos[index + 1]]))
    print(f"   Plateau mean intensities: {means}")
    print(f"Plateau standard deviations: {stdevs}")

    # Plug values into equation and calculate aSNR
    aSNR = 0
    for index in range(len(means) - 1):
        height = abs(means[index] - means[index + 1])
        partial = (2 * height) / (stdevs[index] + stdevs[index + 1])
        aSNR += partial
    aSNR = aSNR / len(means)

    return round(aSNR, 4)
