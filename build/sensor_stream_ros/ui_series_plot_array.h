/********************************************************************************
** Form generated from reading UI file 'series_plot_array.ui'
**
** Created by: Qt User Interface Compiler version 5.12.8
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_SERIES_PLOT_ARRAY_H
#define UI_SERIES_PLOT_ARRAY_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QToolButton>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_SeriesPlotArray
{
public:
    QVBoxLayout *verticalLayout;
    QHBoxLayout *horizontalLayout;
    QComboBox *series;
    QToolButton *add_plot;
    QSpacerItem *horizontalSpacer;
    QVBoxLayout *plot_layout;

    void setupUi(QWidget *SeriesPlotArray)
    {
        if (SeriesPlotArray->objectName().isEmpty())
            SeriesPlotArray->setObjectName(QString::fromUtf8("SeriesPlotArray"));
        SeriesPlotArray->resize(998, 578);
        verticalLayout = new QVBoxLayout(SeriesPlotArray);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        series = new QComboBox(SeriesPlotArray);
        series->setObjectName(QString::fromUtf8("series"));

        horizontalLayout->addWidget(series);

        add_plot = new QToolButton(SeriesPlotArray);
        add_plot->setObjectName(QString::fromUtf8("add_plot"));

        horizontalLayout->addWidget(add_plot);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer);


        verticalLayout->addLayout(horizontalLayout);

        plot_layout = new QVBoxLayout();
        plot_layout->setObjectName(QString::fromUtf8("plot_layout"));

        verticalLayout->addLayout(plot_layout);


        retranslateUi(SeriesPlotArray);

        QMetaObject::connectSlotsByName(SeriesPlotArray);
    } // setupUi

    void retranslateUi(QWidget *SeriesPlotArray)
    {
        SeriesPlotArray->setWindowTitle(QApplication::translate("SeriesPlotArray", "Form", nullptr));
        add_plot->setText(QApplication::translate("SeriesPlotArray", "+", "add plot"));
    } // retranslateUi

};

namespace Ui {
    class SeriesPlotArray: public Ui_SeriesPlotArray {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_SERIES_PLOT_ARRAY_H
