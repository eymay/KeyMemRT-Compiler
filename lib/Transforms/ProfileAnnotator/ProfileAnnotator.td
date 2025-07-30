#ifndef LIB_TRANSFORMS_PROFILEANNOTATOR_PROFILEANNOTATOR_TD_
#define LIB_TRANSFORMS_PROFILEANNOTATOR_PROFILEANNOTATOR_TD_

include "mlir/Pass/PassBase.td"

def ProfileAnnotator : Pass<"profile-annotator"> {
  let summary = "Annotate operations with profile results";
  let description = [{
    This pass reads profile data from runtime execution and annotates
    operations with the result_towers attribute which contains the
    number of towers in the result ciphertext.

    The pass reads profile data in the format:
    PROFILE::OUTPUT:mlir_name:LEVEL:level:NOISE:noise:TOWERS:towers:LINE:line

    Only the TOWERS field is currently used for annotation.
  }];
  let options = [
    Option<"profileFile", "profile-file", "std::string",
           /*default=*/"\"profile.txt\"", "Path to the profile data file">,
  ];
}


#endif  // LIB_TRANSFORMS_PROFILEANNOTATOR_PROFILEANNOTATOR_TD_
