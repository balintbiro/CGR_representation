# import the necessary library
library(optparse)

# make container for CLI arguments
option_list=list(
  make_option(
    c("-e","--encoding"),
    type="character",
    default=NULL,
    help="Order of the 20 proteinogenic amino acids",
    metavar="character"
  ),
  make_option(
      c("-s","--scaling_factor"),
      type="double",
      default=NULL,
      help="Scaling factor for generating the CGR",
      metavar="double"
  ),
  make_option(
      c("-r","--resolution"),
      type="integer",
      default=NULL,
      help="Resolution (projected grid dimension) for FCGR",
      metavar="integer"
  ),
  make_option(
    c("-o","--output_file"),
    type="character",
    default=NULL,
    help="Output file of the FCGR matrix",
    metavar="character"
  ),
  make_option(
    c("-i","--input_filename"),
    type="character",
    default=NULL,
    help="Input filename of the sequences. The file should contain sequence and label columns!",
    metavar="character"
  )
);

#parse CLI arguments
opt_parser=OptionParser(option_list=option_list);
opt=parse_args(opt_parser);

#read in the sequences from the input file
sequences<-read.csv(
  file=opt$input_filename
)
#for testing purposes
#sequences<-sequences[sample(nrow(sequences),10),]

# add indices to the sequences data frame
sequences$index<-seq.int(0,dim(sequences)[1]-1)
################################################
distr.pts = function(
  n,
  r
){

  #get coordinates for a regular polygon
  # this function is encoding-specific as it is used to calculate the coordinates for the bases in an order determined by the encoding
  x = vector("double", n)
  y = vector("double", n)
  for (i in 1:n){
    x[i] = r*sinpi((2*i+1)/n)
    y[i] = r*cospi((2*i+1)/n)
  }
  #return coordinates
  return(xy.coords(x, y))
}
################################################
cgr = function(
  sequence,
  scaling_factor,
  resolution,
  base#coordinates for the monomers around a circle according to the encoding
){

  r = 1
  sequence=strsplit(sequence,"")[[1]]

  #get the length of sequence
  sequence.length = length(sequence)

  #cgr algorithm:
  #start at point (0,0)
  #1. check next character
  #2. go a fraction of the way to the corresponding base, according to the
  #scaling factor
  #3. save coordinates of the point
  #repeat
  x = vector("double", sequence.length)
  y = vector("double", sequence.length)
  A = matrix(data = 0, ncol = resolution, nrow = resolution)
  pt = vector("double", 2)
  for (i in 1:sequence.length) {
    pt = pt + (unlist(base[sequence[i],]) - pt) * scaling_factor
    x[i] = pt[1]
    y[i] = pt[2]
    x.matrix = ceiling((x[i]+r ) * resolution/(2*r))
    y.matrix = ceiling((y[i]+r ) * resolution/(2*r))
    A[x.matrix, y.matrix] = A[x.matrix, y.matrix] + 1
  }
  return(as.vector(t(A)))
}

seq.base=strsplit(opt$encoding,"")[[1]]
#get the number of monomers (amino acids or nucleotides)
base.num = length(seq.base)
#calculate coordinates for the monomers (amino acids or nucleotides) according to the encoding
base.coord = distr.pts(base.num, 1)

#data frame for easy access
#this contains the coordinates for the monomers
#so 20 coordinates around a circle
base = data.frame(x = base.coord$x,
                  y = base.coord$y,
                  row.names = seq.base)

fcgr_list <- lapply(1:nrow(sequences), function(i) {
  cgr(
    sequence = sequences$sequence[i],
    scaling_factor = opt$scaling_factor,
    resolution = opt$resolution,
    base = base
  )
})

fcgr_matrix <- as.data.frame(do.call(rbind, fcgr_list))
fcgr_matrix$label <- sequences$label

output_file <- file.path(opt$output_file)

write.table(
  fcgr_matrix,
  file = output_file,
  row.names = FALSE,
  col.names = TRUE,
  sep = ","
)