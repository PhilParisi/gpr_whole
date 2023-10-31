#ifndef BLOCK_TILER_H
#define BLOCK_TILER_H

#include <vector>
#include <memory>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <sensor_stream/blockgpr.h>
#include <sensor_msgs/PointCloud2.h>
#include <mutex>
#include <include/sensor_stream_ros/common/single_frame_block.h>
namespace ss{namespace ros {


//    struct TrainingBlock{
//        CudaMat<float> x;
//        CudaMat<float> y;
//    };


//    struct BlockParams{
//        size_t size;
//    };

//    typedef std::shared_ptr<BlockParams> BlockParamsPtr;

//    /*!
//     * \brief The Block class represents a block of points that you want to add to a GPR. It has both a
//     * pcl point cloud and a CudaMat<float> representation.
//     */
//    class Block{
//    public:
//        Block();
//        Block(BlockParamsPtr params);
//        Block(BlockParamsPtr params,pcl::PointCloud<pcl::PointXYZI>::Ptr cloud);
//        void setPointcloud(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud);
//        void push_back(pcl::PointXYZI pt);
//        void computeCenterOfMass();
//        void setBlockParam(std::shared_ptr<BlockParams> param){_params=param;}
//        std::shared_ptr<BlockParams> getBlockParam();
//        pcl::PointXYZ getCenterOfMass();
//        size_t size(){return _cloud->size();}
//        pcl::PointCloud<pcl::PointXYZI>::Ptr getCloud(){return _cloud;}
//        void clear();

//        TrainingBlock getTrainingData();


//    private:
//        pcl::PointXYZ centerOfMass;
//        std::shared_ptr<BlockParams> _params;
//        pcl::PointCloud<pcl::PointXYZI>::Ptr _cloud;
//    };


    struct TileParams{
        float xdim; ///< the size of the tile in meters;
        float ydim; ///< the size of the tile in meters;
        float zdim; ///< currently not implemeted
        float xMargin;
        float yMargin;
        float zMargin;
    };

    typedef std::shared_ptr<TileParams> TileParamsPtr;

    struct TileData{
        //! \todo add mutex for later multithreading
        bool needsProcessing;///< means that the tile is in a queue to be processed
        bool visited;///< means that the tile has already been visited in some kind of search
        bool inTiler;///< indicates weather or not this tile has been added to a BlockTiler
        float xOrigin;///< origin at the CENTER of the tile
        float yOrigin;///< origin at the CENTER of the tile
        size_t cholNnz;
        sensor_msgs::PointCloud2::Ptr prediction;

        TileData(){
            needsProcessing = false;
            visited = false;
            inTiler = false;
            prediction = NULL;
            cholNnz = 0;
        }
        std::mutex mutex;
    };

    typedef std::shared_ptr<TileData> TileDataPtr;

    /*!
     * \brief The Tile class is a container for a set of blocks
     * this class is headder class for heap memory and therefore must
     * be explicitly deep copied if desired.
     */
    class Tile{

    private:
        std::shared_ptr< std::vector<SingleFrameBlock> > _blocks;  ///< the vector of blocks that the tile contains
        /*! \brief the parameters that define the dimentions of the tile
            \details this data memeber can be shared between tiles.   */
        TileParamsPtr  _params;
        TileDataPtr _data;  ///< metadata about the tile see ss::ros::TileData
    public:

        Tile();
        Tile(std::shared_ptr<TileParams> params);
        void setTileParam(std::shared_ptr<TileParams > params){_params=params;}
        void addBlock(SingleFrameBlock block);
        SingleFrameBlock & getBlock(size_t i){return _blocks->operator [](i);}
        size_t size(){return _blocks->size();}
        float getAvgZ();

        pcl::PointCloud<pcl::PointXYZI>::Ptr predict();  ///< \brief make a prediction using the latest available training data

        bool isVisited(){return _data->visited;}
        void setVisited(bool x){_data->visited = x;}

        bool isQueued(){return _data->needsProcessing;}
        void setQueued(bool x){_data->needsProcessing = x;}

        bool inTiler(){return _data->inTiler;}
        void setInTiler(bool x){_data->inTiler = x;}

        void setPrediction(sensor_msgs::PointCloud2::Ptr pred){_data->mutex.lock(); _data->prediction =  pred; _data->mutex.unlock();}
        sensor_msgs::PointCloud2::Ptr getPrediction(){return _data->prediction;}

        void setCenter(float x,float y){_data->xOrigin=x;_data->yOrigin=y;}
        TileDataPtr getData(){return _data;}
        //TileData getMetadata(){return *_data;}
        TileParams getTileParam(){return *_params;}

    };



    template<class T>
    class doubleVector{
    public:
        doubleVector(){return;}
        /*!
         * \brief inserts an object at the specified index.  The vector will automatically expand if the index is out of range.
         * \param item the item you wish to insert
         * \param i the index you want to insert at.  may be negative.
         */
        void insert(T item, int32_t i);
        T & operator[](int32_t i);
    private:
        std::vector<T> _positiveVect;
        std::vector<T> _negativeVect;

    };

    /*!
     * \brief This class is designed to organize a set of GPR Data Blocks (roughy) into an even grid (known as tiles) for fast lookup.
     *        It should be noted that blocks fall into tiles based on there "center of mass" (average postition) so it is possible
     *        for blocks to spill over into neighboring tiles.
     */
    class BlockTiler
    {
    protected:
        TileParamsPtr  _tileParams;
        BlockParams::Ptr _blockParams;
        doubleVector<doubleVector<Tile> > _grid;
        std::vector<Tile> _tileList;
        void tileUpdated(Tile tile, SingleFrameBlock block);
    public:
        BlockTiler();
        BlockTiler(TileParamsPtr tileParams,BlockParams::Ptr blockParams){
            setTileParams(tileParams);setBlockParams(blockParams);}
        /*!
         * \brief adds a block to all relevant tiles.
         * A relevant tile one who's boundaries + margins contain the center of
         * mass of the block being added.
         * \param block [input] the block you wan to add
         * \return a vector of updated tiles
         */
        std::vector<Tile> addBlock(SingleFrameBlock block);
        Tile & getTile(float x, float y);
        void addToTileList(Tile tile);
        std::vector<Tile> & getTileList(){return _tileList;}
        Tile tileList(size_t i){return _tileList[i];}
        void setTileParams(std::shared_ptr<TileParams> tileParams){_tileParams=tileParams;}
        const TileParamsPtr getTileParam(){return _tileParams;}
        void setBlockParams(BlockParams::Ptr blockParams){_blockParams=blockParams;}

    };

    typedef std::shared_ptr<BlockTiler> BlockTilerPtr;



}} // end namespace def



// define template stuff down here

template<class T>
void ss::ros::doubleVector<T>::insert(T item, int32_t i){
    if(i>=0){
        if(i>=_positiveVect.size()){
            _positiveVect.resize(i+1);
        }
        _positiveVect[i]=item;
    }else{
        i=abs(i);
        if(i>=_negativeVect.size()){
            _negativeVect.resize(i+1);
        }
        _negativeVect[i]=item;
    }
    return;
}

template<class T>
T & ss::ros::doubleVector<T>::operator[](int32_t i){
    T output;
    // expand if necessary
    if(i>=0){
        if(i>=_positiveVect.size()){
            _positiveVect.resize(i+1);
        }
        return _positiveVect[i];
    }else{
        i=abs(i);
        if(i>=_negativeVect.size()){
            _negativeVect.resize(i+1);
        }
        return _negativeVect[i];
    }
}


#endif // BLOCK_TILER_H
