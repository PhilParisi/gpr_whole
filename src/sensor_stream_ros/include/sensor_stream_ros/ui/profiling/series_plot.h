#ifndef SERIES_PLOT_H
#define SERIES_PLOT_H

#include <QWidget>
#include <sensor_stream_ros/profiling/series.h>
#include <sensor_stream_ros/third_party/qcustomplot.h>
#include <unordered_map>
#include <boost/bimap.hpp>

namespace Ui {
class SeriesPlot;
}
namespace graph_type {
enum GraphType{
  value,
  upper,
  lower
};
static const char * strings[] = {  "value", "upper", "lower"};
}
struct GraphKey{
  GraphKey(std::string x, std::string y ,graph_type::GraphType t = graph_type::value){
    x_key=x; y_key=y; type = t;
  }
  GraphKey getUpper()const{GraphKey out = *this; out.type = graph_type::upper; return out;}
  GraphKey getLower()const{GraphKey out = *this; out.type = graph_type::lower; return out;}
  GraphKey getValue()const{GraphKey out = *this; out.type = graph_type::value; return out;}
  std::string  toString() const {return x_key+"|"+y_key+"|"+graph_type::strings[type];}
  bool operator <(const GraphKey & other) const {return toString() < other.toString();}
  std::string x_key;
  std::string y_key;
  graph_type::GraphType type;
};

struct GraphWithIterator{
  GraphWithIterator(QCPGraph* graph_in){
    graph = graph_in;
    last_index.reset(new size_t(0));
  }
  bool operator <(const GraphWithIterator & other) const {return graph < other.graph;}
  QCPGraph* graph;
  std::shared_ptr<size_t> last_index;
};

class SeriesPlot : public QCustomPlot
{
  Q_OBJECT

public:
  explicit SeriesPlot(QWidget *parent = nullptr);
  ~SeriesPlot();

  void setSeries(ss::profiling::Series::Ptr series){series_ptr_=series;}
  void addGraph(std::string x_key,std::string y_key);
  void addErrorBars(GraphKey key);
  bool hasGraph(std::string x_key,std::string y_key);
  bool hasGraph(GraphKey key);
  bool hasErrorBars(GraphKey key);
  QCPGraph *graph(GraphKey key);
  size_t & lastIndex(GraphKey key);
  GraphKey key(QCPGraph* graph);
  boost::bimap<GraphKey,GraphWithIterator> getKeyMap(){return graph_key_map_;}
  void setColor(GraphKey key, QColor color);
  void setLineStyle(GraphKey key, QCPGraph::LineStyle style);

public slots:
  void update();
  void removeGraph(GraphKey key);


private:
  Ui::SeriesPlot *ui;
  ss::profiling::Series::Ptr series_ptr_;
  //std::vector<GraphKey> graph_keys_;
  boost::bimap<GraphKey,GraphWithIterator> graph_key_map_;
  //std::map<GraphKey,int> var_key_map_;
};

#endif // SERIES_PLOT_H
