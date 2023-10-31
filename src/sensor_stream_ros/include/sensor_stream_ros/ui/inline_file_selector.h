#ifndef INLINE_FILE_SELECTOR_H
#define INLINE_FILE_SELECTOR_H

#include <QWidget>

namespace Ui {
class InlineFileSelector;
}

class InlineFileSelector : public QWidget
{
  Q_OBJECT

public:
  explicit InlineFileSelector(QWidget *parent = nullptr);
  ~InlineFileSelector();
  void setFileType(QString filetype);
  void setText(QString text);
  QString getFile();

private slots:
  //void on_browse_button_clicked();

private:
  Ui::InlineFileSelector *ui;
};

#endif // INLINE_FILE_SELECTOR_H
