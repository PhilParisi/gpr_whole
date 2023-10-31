/********************************************************************************
** Form generated from reading UI file 'series_plot.ui'
**
** Created by: Qt User Interface Compiler version 5.12.8
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_SERIES_PLOT_H
#define UI_SERIES_PLOT_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>
#include "include/sensor_stream_ros/third_party/qcustomplot.h"

QT_BEGIN_NAMESPACE

class Ui_SeriesPlot
{
public:
    QVBoxLayout *verticalLayout;
    QCustomPlot *plot;

    void setupUi(QWidget *SeriesPlot)
    {
        if (SeriesPlot->objectName().isEmpty())
            SeriesPlot->setObjectName(QString::fromUtf8("SeriesPlot"));
        SeriesPlot->resize(400, 300);
        verticalLayout = new QVBoxLayout(SeriesPlot);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        plot = new QCustomPlot(SeriesPlot);
        plot->setObjectName(QString::fromUtf8("plot"));

        verticalLayout->addWidget(plot);


        retranslateUi(SeriesPlot);

        QMetaObject::connectSlotsByName(SeriesPlot);
    } // setupUi

    void retranslateUi(QWidget *SeriesPlot)
    {
        SeriesPlot->setWindowTitle(QApplication::translate("SeriesPlot", "Form", nullptr));
    } // retranslateUi

};

namespace Ui {
    class SeriesPlot: public Ui_SeriesPlot {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_SERIES_PLOT_H
