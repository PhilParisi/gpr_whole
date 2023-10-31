#include "block_tiler.h"


//ss::ros::Block::Block(){
//    clear();
//}


//ss::ros::Block::Block(std::shared_ptr<BlockParams> params){
//    setBlockParam(params);
//    _cloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
//}

//ss::ros::Block::Block(BlockParamsPtr params,pcl::PointCloud<pcl::PointXYZI>::Ptr cloud){
//    setBlockParam(params);
//    setPointcloud(cloud);
//}

//void ss::ros::Block::clear(){
//    _params = nullptr;
//    _cloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
//    centerOfMass.x += 0;
//    centerOfMass.y += 0;
//    centerOfMass.z += 0;
//}

//void ss::ros::Block::setPointcloud(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud){
//    size_t cloud_size = cloud->width * cloud->height;
//    if(cloud_size!=getBlockParam()->size)
//        throw std::out_of_range("setPointcloud error:  cloud size != BlockParams size");

//    _cloud=cloud;
//    computeCenterOfMass();

//    return;
//}

//void ss::ros::Block::push_back(pcl::PointXYZI pt){
//    _cloud->push_back(pt);
//    centerOfMass.x += pt.x;
//    centerOfMass.y += pt.y;
//    centerOfMass.z += pt.z;
//}

//void ss::ros::Block::computeCenterOfMass(){
//    size_t cloud_size = _cloud->width * _cloud->height;
//    float x = 0;
//    float y = 0;
//    float z = 0;
//    for ( size_t i = 0; i < cloud_size; i++ )
//    {
//        x += _cloud->at(i).x;
//        y += _cloud->at(i).y;
//        z += _cloud->at(i).z;
//    }
//    centerOfMass.x = x;
//    centerOfMass.y = y;
//    centerOfMass.z = z;
//}

//pcl::PointXYZ ss::ros::Block::getCenterOfMass(){
//    size_t cloud_size = _cloud->width * _cloud->height;
//    pcl::PointXYZ out;
//    out.x = centerOfMass.x/cloud_size;
//    out.y = centerOfMass.y/cloud_size;
//    out.z = centerOfMass.z/cloud_size;
//    return out;
//}

//std::shared_ptr<ss::ros::BlockParams> ss::ros::Block::getBlockParam(){
//    if(_params==NULL){
//        throw std::out_of_range("No BlockParams has been set!  Call setBlockParam first or use the Block(std::shared_ptr<BlockParams> params) constructor");
//    }
//    return _params;
//}

//ss::ros::TrainingBlock ss::ros::Block::getTrainingData(){
//    TrainingBlock block;

//    block.x.reset(_params->size,2);
//    block.x.initHost();

//    block.y.reset(_params->size,1);
//    block.y.initHost();

//    size_t i = 0;
//    for (pcl::PointCloud<pcl::PointXYZI>::const_iterator it = _cloud->begin(); it != _cloud->end(); ++it){
//        block.x(i,0) = it->x;
//        block.x(i,1) = it->y;
//        block.y(i,0) = it->z;
//        i++;
//    }

//    block.x.host2dev();
//    block.y.host2dev();

//    return block;
//}

ss::ros::Tile::Tile(){
    _params = NULL;
    _blocks.reset(new std::vector<SingleFrameBlock> );
    _data.reset(new TileData);
}
ss::ros::Tile::Tile(std::shared_ptr<TileParams> params){
    _params=params;
    _blocks.reset(new std::vector<SingleFrameBlock> );
    _data.reset(new TileData);
}


void ss::ros::Tile::addBlock(SingleFrameBlock block){
    _blocks->push_back(block);
}

float ss::ros::Tile::getAvgZ(){
  float avg = 0;
  for (size_t i = 0;i<_blocks->size();i++) {
    avg+=_blocks->operator[](i).getCenterOfMass().z;
  }
  avg = avg/_blocks->size();
  return avg;
}


ss::ros::BlockTiler::BlockTiler(){

}

std::vector<ss::ros::Tile> ss::ros::BlockTiler::addBlock(SingleFrameBlock block){
  block.setParams(_blockParams);
  std::vector<Tile> visitedTiles;
  for(int i = -1; i<=1 ; i++){
    for(int j = -1; j<=1 ; j++){
      if(std::isnan(block.getCenterOfMass().x)||std::isnan(block.getCenterOfMass().y)||std::isnan(block.getCenterOfMass().z)){
        std::cerr << "got NAN block center of mass" << std::endl;
      }else{
        float xIndex = block.getCenterOfMass().x + i*_tileParams->xMargin;
        float yIndex = block.getCenterOfMass().y + j*_tileParams->yMargin;
        Tile newTile = getTile(xIndex,yIndex);
        if(!newTile.isVisited()){
          newTile.addBlock(block);
          newTile.setTileParam(_tileParams);
          newTile.setVisited(true);
          //*newTile.visited=true;
          visitedTiles.push_back(newTile);
        }
      }
    }
  }

  for(size_t i = 0 ; i<visitedTiles.size() ; i++){
    visitedTiles[i].setVisited(false);
    //*visitedTiles[i].visited=false;
    tileUpdated(visitedTiles[i],block);
  }

  return visitedTiles;
}

void ss::ros::BlockTiler::tileUpdated(Tile tile, SingleFrameBlock block){

}

ss::ros::Tile & ss::ros::BlockTiler::getTile(float x, float y){
    int32_t xindex;
    int32_t yindex;
    float xCenter;
    float yCenter;

    if(x>=0){
         xindex=int32_t(x/_tileParams->xdim)+1;
    }else{
         xindex=int32_t(x/_tileParams->xdim)-1;
    }
    if(y>=0){
         yindex=int32_t(y/_tileParams->ydim)+1;
    }else {
         yindex=int32_t(y/_tileParams->ydim)-1;
    }

    xCenter = _tileParams->xdim * xindex - xindex/fabs(xindex) * _tileParams->xdim/2.0f;
    yCenter = _tileParams->ydim * yindex - yindex/fabs(yindex) * _tileParams->ydim/2.0f;

    _grid[xindex][yindex].setCenter(xCenter,yCenter);

    addToTileList(_grid[xindex][yindex]);
    return _grid[xindex][yindex];

}

void ss::ros::BlockTiler::addToTileList(Tile tile){
  if(!tile.inTiler()){
    tile.setInTiler(true);
    _tileList.push_back(tile);
  }
  return;
}


