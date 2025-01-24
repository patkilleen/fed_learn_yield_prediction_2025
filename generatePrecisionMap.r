library(sp)
library(sf)
library(raster)
library(ggplot2)
library(shape) #for colorlegend


#inputDatasetPath: path to yield dataset, which expects X,Y,yield columns
#outputImagePath: path to output precision map image
#bgdImgPath: path to the background raster image overlaid behind the precisin map and field boundaries
#mapTitle: title of the map
#smallFieldsFlag: boolean flag indicating True when its a map of the small fields, and False when its a map of Field 11 (the big field)
#field_boundary_file_paths: list of field boundary shapefile file paths for creating field borders on the map
#myCRS: coordinate reference string 
createYieldPrecisionMap=function(inputDatasetPath,outputImagePath, bgdImgPath, mapTitle,smallFieldsFlag,field_boundary_file_paths,myCRS,legendTitle){
	require(sp)
	require(sf)
	require(raster)
	require(ggplot2)
	require(shape) 
	

	#read the yield dataset into memory
	y_df = read.csv(inputDatasetPath,sep=",")
	
	#compute the quartiles for determine appropriate bin ranges for colors (so outliers avoid skewing bin color ranges)
	quantiles = quantile(y_df$yield)
	q1 = quantiles[2]
	q3 = quantiles[4]
	irq = q3 - q1
	
	#max and min color ranges for the legend
	maxVal = as.numeric(q3+irq*1.1)
	minVal = as.numeric(max(q1-irq*1.1,0))
	
	#can comment above two lines, and uncomment below two lines, to force/hardcode the legend's min and max bin values to force multiple maps to have the same legend scale
	#maxVal=202
	#minVal=60
	
	ix_above_maxVal = which(y_df$yield > maxVal)
	ix_above_minVal = which(y_df$yield < minVal)
	
	#make sure to replace extreme yield outliers with the colors legend ranges
	y_df$yield[ix_above_maxVal] = maxVal
	y_df$yield[ix_above_minVal] = minVal
	
	#10 colors from red to yellow to green
	numberOfColorBins = 10
	
	binSize = (maxVal-minVal)/numberOfColorBins
	red_yellow_green <- colorRampPalette(c("red", "yellow", "green"))
	colorBins = red_yellow_green(numberOfColorBins)
	yColors <- colorBins[as.numeric(cut(y_df$yield,breaks = seq(from = minVal-0.1, to =maxVal, length.out =numberOfColorBins)))] 
	y_df = as.data.frame(cbind(y_df$X,y_df$Y))
	colnames(y_df)=c("X","Y")
	
	utm_crs = myCRS
	new_sf = st_as_sf(y_df,coords=c("X","Y"),crs=utm_crs)

	
	#read the background raster image
	ras = brick(bgdImgPath)

	#crop the raster to fit nicely to the yield dataset spatial extent
	bb = extent(new_sf)

	#adjust the margins of the background raster based on wether big or small fields are involved
	if(smallFieldsFlag){
		#smaller field margings
		attr(bb,"xmin") = attr(bb,"xmin")- 35
		attr(bb,"ymin") = attr(bb,"ymin")- 15
		attr(bb,"ymax") = attr(bb,"ymax")+ 15
		attr(bb,"xmax") = attr(bb,"xmax")+ 35
	}else{
		#filed 11 margins
		attr(bb,"xmin") = attr(bb,"xmin")- 85
		attr(bb,"ymin") = attr(bb,"ymin")- 150
		attr(bb,"ymax") = attr(bb,"ymax")+ 40
		attr(bb,"xmax") = attr(bb,"xmax")+ 100
	}

	cras = crop(ras,bb)
	
	#draw the background raster
	png(outputImagePath,width=2,height=2,units="in",res=1200,pointsize=3,type = "cairo-png")
	plotRGB(cras,main=mapTitle,margins=c(50,50,50,50))



	#create the precision map
	plot(new_sf,pch=19,cex = 0.75,col=yColors,add=TRUE,lwd=0.0025)


	#draw the field boundaries
	for (path_to_boundary_shapefile in field_boundary_file_paths){
		
		#read field boundary shapefile
		boundary_sp = st_read(path_to_boundary_shapefile)

		#convert to UTM
		boundary_sp.utm = st_transform(boundary_sp$geometry,crs = myCRS)
		boundary_sp$geometry = boundary_sp.utm
		#plot each field's boundary		
		plot(boundary_sp, col=rgb(0.5,0.5,0.5,alpha=0.15),lwd=0.8, add=TRUE)
	}

	#draw the legend
	colorlegend(left = FALSE,#text is left of the legend
		cex=1.25,
		main.cex=1.25,
		digit = 0, #number of significant digits		
		dz = binSize,#increment for legennd bin sizes. size of color bins (how many number to display )
		col = colorBins, zlim = c(minVal, maxVal), 
		zlevels = NULL, main = legendTitle)


	dev.off()#save iamge
}



inputDatasetPath="input/datasets/F11-multi-temporal-IID.csv"
outputImagePath="output/Field11-actual-yield-precision-map.png"
	
bgdImgPath = "raw-data/area-x.o-satellite-image-raster.tif"
	
field_boundary_file_paths=list("raw-data/field-boundaries/Field1/boundary.shp")
field_boundary_file_paths[2]="raw-data/field-boundaries/Field2/boundary.shp"
field_boundary_file_paths[3]="raw-data/field-boundaries/Field3/boundary.shp"
field_boundary_file_paths[4]="raw-data/field-boundaries/Field4/boundary.shp"
field_boundary_file_paths[5]="raw-data/field-boundaries/Field9/boundary.shp"
field_boundary_file_paths[6]="raw-data/field-boundaries/Field11/boundary.shp"
mapTitle="Field 11 Actual Yield"

#this is a hardcoded coordinate reference system string and works for Area X.O but may not for yield datasets in another region.
myCRS="+proj=utm +zone=18 +ellps=WGS84 +datum=WGS84 +units=m +no_defs" 
smallFieldsFlag=FALSE

legendTitle ="Yield \n(bu/ac)"
createYieldPrecisionMap(inputDatasetPath,outputImagePath, bgdImgPath, mapTitle,smallFieldsFlag,field_boundary_file_paths,myCRS,legendTitle)