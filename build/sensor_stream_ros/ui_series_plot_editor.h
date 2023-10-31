/********************************************************************************
** Form generated from reading UI file 'series_plot_editor.ui'
**
** Created by: Qt User Interface Compiler version 5.12.8
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_SERIES_PLOT_EDITOR_H
#define UI_SERIES_PLOT_EDITOR_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QFormLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>
#include "include/sensor_stream_ros/ui/profiling/graph_edit.h"
#include "include/sensor_stream_ros/ui/profiling/series_plot.h"

QT_BEGIN_NAMESPACE

class Ui_SeriesPlotEditor
{
public:
    QHBoxLayout *horizontalLayout;
    SeriesPlot *plot;
    QVBoxLayout *verticalLayout;
    QFormLayout *formLayout_2;
    QLabel *label;
    QComboBox *x_selection;
    QLabel *label_2;
    QComboBox *y_selection;
    QPushButton *add_plot;
    GraphEdit *graph_editor;
    QSpacerItem *verticalSpacer;

    void setupUi(QWidget *SeriesPlotEditor)
    {
        if (SeriesPlotEditor->objectName().isEmpty())
            SeriesPlotEditor->setObjectName(QString::fromUtf8("SeriesPlotEditor"));
        SeriesPlotEditor->resize(1356, 732);
        horizontalLayout = new QHBoxLayout(SeriesPlotEditor);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        plot = new SeriesPlot(SeriesPlotEditor);
        plot->setObjectName(QString::fromUtf8("plot"));
        QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(plot->sizePolicy().hasHeightForWidth());
        plot->setSizePolicy(sizePolicy);

        horizontalLayout->addWidget(plot);

        verticalLayout = new QVBoxLayout();
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        formLayout_2 = new QFormLayout();
        formLayout_2->setObjectName(QString::fromUtf8("formLayout_2"));
        label = new QLabel(SeriesPlotEditor);
        label->setObjectName(QString::fromUtf8("label"));

        formLayout_2->setWidget(0, QFormLayout::LabelRole, label);

        x_selection = new QComboBox(SeriesPlotEditor);
        x_selection->setObjectName(QString::fromUtf8("x_selection"));

        formLayout_2->setWidget(0, QFormLayout::FieldRole, x_selection);

        label_2 = new QLabel(SeriesPlotEditor);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        formLayout_2->setWidget(1, QFormLayout::LabelRole, label_2);

        y_selection = new QComboBox(SeriesPlotEditor);
        y_selection->setObjectName(QString::fromUtf8("y_selection"));

        formLayout_2->setWidget(1, QFormLayout::FieldRole, y_selection);


        verticalLayout->addLayout(formLayout_2);

        add_plot = new QPushButton(SeriesPlotEditor);
        add_plot->setObjectName(QString::fromUtf8("add_plot"));

        verticalLayout->addWidget(add_plot);

        graph_editor = new GraphEdit(SeriesPlotEditor);
        graph_editor->setObjectName(QString::fromUtf8("graph_editor"));

        verticalLayout->addWidget(graph_editor);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout->addItem(verticalSpacer);


        horizontalLayout->addLayout(verticalLayout);


        retranslateUi(SeriesPlotEditor);

        QMetaObject::connectSlotsByName(SeriesPlotEditor);
    } // setupUi

    void retranslateUi(QWidget *SeriesPlotEditor)
    {
        SeriesPlotEditor->setWindowTitle(QApplication::translate("SeriesPlotEditor", "Form", nullptr));
        label->setText(QApplication::translate("SeriesPlotEditor", "X-Axis", nullptr));
        label_2->setText(QApplication::translate("SeriesPlotEditor", "Y-axis", nullptr));
        add_plot->setText(QApplication::translate("SeriesPlotEditor", "Add", nullptr));
    } // retranslateUi

};

namespace Ui {
    class SeriesPlotEditor: public Ui_SeriesPlotEditor {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_SERIES_PLOT_EDITOR_H
