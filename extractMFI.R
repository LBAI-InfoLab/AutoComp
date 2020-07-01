


##
## Part of the compensation tool
##



## loading packages
suppressPackageStartupMessages(
  {
    library(flowCore)
  })


Extract_MFI_Information <- function(file_path){
  
  ##
  ## Extract Compensation matrix
  ## From an uncompensated FCS file
  ##
  
  ## Open FCS file
  flowDataPath = file_path
  flowData <-read.FCS(flowDataPath,transformation = FALSE, alter.names = FALSE,emptyValue=FALSE)
  
  ## Extract informations on MFI values
  MFI_informations <- summary(flowData)
  
  ## Save Informations in a csv file
  file_in_array <- strsplit(file_path, "/")
  file_in_array <- file_in_array[[1]][length(file_in_array[[1]])]
  file_in_array <- strsplit(file_in_array, "_")
  panel <- file_in_array[[1]][2]
  patientID <- file_in_array[[1]][3]
  center <- file_in_array[[1]][4]
  
  
  ## Check if the input is compensated or uncompensated, determine
  ## the outout folder and filename
  if(length(file_in_array[[1]]) == 9){
    output_folder <- "/home/glorfindel/Spellcraft/SIDEQUEST/compensation/data/MFI/compensated/"
    output_filename <- paste(output_folder,"Panel_",panel,"_",center,"_",patientID,"_compensated.txt", sep ="")
  }else{
    output_folder <- "/home/glorfindel/Spellcraft/SIDEQUEST/compensation/data/MFI/uncompensated/"
    output_filename <- paste(output_folder,"Panel_",panel,"_",center,"_",patientID,"_uncompensated.txt", sep ="")
  }
  
  ## Write MFI information in a csv file
  write.csv(MFI_informations, output_filename)
}


## Run the function
args <- commandArgs(trailingOnly = TRUE)
file_path<-args[1]
#file_path = "/home/glorfindel/Spellcraft/SIDEQUEST/compensation/Panel_5_42_TEST_blablabla.fcs"
#file_path = "/home/glorfindel/Spellcraft/SIDEQUEST/compensation/Panel_5_42_TEST_blablabla.fcs"
#file_path = "/home/glorfindel/Spellcraft/SIDEQUEST/compensation/data/fcs/compensated/Panel_1_32161128_UBO_NAVIOS_06JUL2017_06JUL2017.LMD_intra.fcs_comp.fcs"
Extract_MFI_Information(file_path)











