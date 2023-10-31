#include "inline_file_selector.h"
#include "ui_inline_file_selector.h"

InlineFileSelector::InlineFileSelector(QWidget *parent) :
  QWidget(parent),
  ui(new Ui::InlineFileSelector)
{
  ui->setupUi(this);
}

InlineFileSelector::~InlineFileSelector()
{
  delete ui;
}

//void InlineFileSelector::setText(QString text)
//  ui->label->setText(text);
//}

//QString InlineFileSelector::getFile(){
//  return ui->browse_button->text();
//}

//void InlineFileSelector::on_browse_button_clicked()
//{
//  QString fileName = QFileDialog::getOpenFileName(this, tr("Specify A Bag File"),
//                                                  "/home",
//                                                  tr("Bag File (*.bag)"));
//  ui->browse_button->setText(fileName);
//}
