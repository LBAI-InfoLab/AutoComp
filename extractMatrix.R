


##
## Part of the compensation tool
##



## loading packages
suppressPackageStartupMessages(
  {
    library(flowCore)
  })


extractCompensationMatrix <- function(file_path){
  
  ##
  ## Extract Compensation matrix
  ## From an uncompensated FCS file
  ##
  
  ## Open FCS file
  flowDataPath = file_path
  flowData <-read.FCS(flowDataPath,transformation = FALSE, alter.names = FALSE,emptyValue=FALSE)
  
  ## Extract Matrix
  ## Sometimes stored in SPILLOVER, sometimes not ...
  compensation_matrix <- description(flowData)$`$SPILLOVER`
  if(is.null(compensation_matrix)) {
    compensation_matrix <- description(flowData)$SPILL
  }
  
  
  ## write matrix in a csv file
  file_in_array <- strsplit(file_path, "/")
  file_in_array <- file_in_array[[1]][length(file_in_array[[1]])]
  file_in_array <- strsplit(file_in_array, "_")
  panel <- file_in_array[[1]][2]
  patientID <- file_in_array[[1]][3]
  center <- file_in_array[[1]][4]
  
  
  ## Check if the input is compensated or uncompensated, determine
  ## the outout folder and filename
  if(length(file_in_array[[1]]) == 9){
    output_folder <- "/home/glorfindel/Spellcraft/SIDEQUEST/compensation/data/matrix/compensated/"
    output_filename <- paste(output_folder,"Panel_",panel,"_",center,"_",patientID,".txt", sep ="")
  }else{
    output_folder <- "/home/glorfindel/Spellcraft/SIDEQUEST/compensation/data/matrix/uncompensated/"
    output_filename <- paste(output_folder,"Panel_",panel,"_",center,"_",patientID,"_uncompensated.txt", sep ="")
  }
  write.csv(compensation_matrix, output_filename)
}


## Run the function
args <- commandArgs(trailingOnly = TRUE)
file_path<-args[1]
#file_path = "/home/glorfindel/Spellcraft/SIDEQUEST/compensation/Panel_5_42_TEST_blablabla.fcs"
#file_path = "/home/glorfindel/Spellcraft/SIDEQUEST/compensation/data/fcs/raw/Panel_8_32141769_IRCCS_CANTOII_09DEC2015_09DEC2015.fcs_norm.fcs"
extractCompensationMatrix(file_path)













