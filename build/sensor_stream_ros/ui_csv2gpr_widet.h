/********************************************************************************
** Form generated from reading UI file 'csv2gpr_widet.ui'
**
** Created by: Qt User Interface Compiler version 5.12.8
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_CSV2GPR_WIDET_H
#define UI_CSV2GPR_WIDET_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QFormLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>
#include <include/sensor_stream_ros/ui/regression_params_widget.h>
#include "include/sensor_stream_ros/third_party/qcustomplot.h"

QT_BEGIN_NAMESPACE

class Ui_CSV2GPRWidet
{
public:
    QVBoxLayout *verticalLayout;
    QPushButton *open_csv_btn;
    RegressionParamsWidget *params_widget;
    QFormLayout *formLayout_3;
    QLabel *label_6;
    QSpinBox *block_size;
    QTabWidget *tabWidget;
    QWidget *tab;
    QVBoxLayout *verticalLayout_2;
    QFormLayout *formLayout;
    QLabel *label;
    QHBoxLayout *horizontalLayout_2;
    QDoubleSpinBox *x_min;
    QDoubleSpinBox *x_max;
    QLabel *label_2;
    QDoubleSpinBox *y_val;
    QLabel *label_3;
    QSpinBox *divisions;
    QPushButton *predict_btn;
    QCustomPlot *pred_plot;
    QWidget *tab_2;
    QVBoxLayout *verticalLayout_3;
    QFormLayout *formLayout_2;
    QLabel *hp_x_axis;
    QHBoxLayout *horizontalLayout_3;
    QLabel *label_4;
    QDoubleSpinBox *hp_x_min;
    QLabel *label_5;
    QDoubleSpinBox *hp_x_max;
    QLabel *hp_x_div_lbl;
    QSpinBox *hp_x_div;
    QLabel *label_7;
    QHBoxLayout *horizontalLayout_4;
    QLabel *hp_y_min_lbl;
    QDoubleSpinBox *hp_y_min;
    QLabel *hp_y_max_lbl;
    QDoubleSpinBox *hp_y_max;
    QLabel *label_10;
    QSpinBox *hp_y_div;
    QPushButton *generate_lml;
    QCustomPlot *lml_plot;
    QHBoxLayout *horizontalLayout;
    QDoubleSpinBox *lml_min;
    QDoubleSpinBox *lml_max;
    QLabel *pos_lbl;

    void setupUi(QWidget *CSV2GPRWidet)
    {
        if (CSV2GPRWidet->objectName().isEmpty())
            CSV2GPRWidet->setObjectName(QString::fromUtf8("CSV2GPRWidet"));
        CSV2GPRWidet->resize(942, 1157);
        verticalLayout = new QVBoxLayout(CSV2GPRWidet);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        open_csv_btn = new QPushButton(CSV2GPRWidet);
        open_csv_btn->setObjectName(QString::fromUtf8("open_csv_btn"));

        verticalLayout->addWidget(open_csv_btn);

        params_widget = new RegressionParamsWidget(CSV2GPRWidet);
        params_widget->setObjectName(QString::fromUtf8("params_widget"));

        verticalLayout->addWidget(params_widget);

        formLayout_3 = new QFormLayout();
        formLayout_3->setObjectName(QString::fromUtf8("formLayout_3"));
        label_6 = new QLabel(CSV2GPRWidet);
        label_6->setObjectName(QString::fromUtf8("label_6"));

        formLayout_3->setWidget(0, QFormLayout::LabelRole, label_6);

        block_size = new QSpinBox(CSV2GPRWidet);
        block_size->setObjectName(QString::fromUtf8("block_size"));
        block_size->setMinimum(1);
        block_size->setMaximum(3200);
        block_size->setValue(400);

        formLayout_3->setWidget(0, QFormLayout::FieldRole, block_size);


        verticalLayout->addLayout(formLayout_3);

        tabWidget = new QTabWidget(CSV2GPRWidet);
        tabWidget->setObjectName(QString::fromUtf8("tabWidget"));
        tab = new QWidget();
        tab->setObjectName(QString::fromUtf8("tab"));
        verticalLayout_2 = new QVBoxLayout(tab);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        formLayout = new QFormLayout();
        formLayout->setObjectName(QString::fromUtf8("formLayout"));
        label = new QLabel(tab);
        label->setObjectName(QString::fromUtf8("label"));

        formLayout->setWidget(0, QFormLayout::LabelRole, label);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        x_min = new QDoubleSpinBox(tab);
        x_min->setObjectName(QString::fromUtf8("x_min"));
        x_min->setMinimum(-999.000000000000000);
        x_min->setMaximum(999.000000000000000);
        x_min->setValue(-8.000000000000000);

        horizontalLayout_2->addWidget(x_min);

        x_max = new QDoubleSpinBox(tab);
        x_max->setObjectName(QString::fromUtf8("x_max"));
        x_max->setValue(8.000000000000000);

        horizontalLayout_2->addWidget(x_max);


        formLayout->setLayout(0, QFormLayout::FieldRole, horizontalLayout_2);

        label_2 = new QLabel(tab);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        formLayout->setWidget(1, QFormLayout::LabelRole, label_2);

        y_val = new QDoubleSpinBox(tab);
        y_val->setObjectName(QString::fromUtf8("y_val"));
        y_val->setMinimum(-9999.000000000000000);
        y_val->setMaximum(9999.000000000000000);

        formLayout->setWidget(1, QFormLayout::FieldRole, y_val);

        label_3 = new QLabel(tab);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        formLayout->setWidget(2, QFormLayout::LabelRole, label_3);

        divisions = new QSpinBox(tab);
        divisions->setObjectName(QString::fromUtf8("divisions"));
        divisions->setMinimum(10);
        divisions->setMaximum(256);
        divisions->setValue(256);

        formLayout->setWidget(2, QFormLayout::FieldRole, divisions);


        verticalLayout_2->addLayout(formLayout);

        predict_btn = new QPushButton(tab);
        predict_btn->setObjectName(QString::fromUtf8("predict_btn"));

        verticalLayout_2->addWidget(predict_btn);

        pred_plot = new QCustomPlot(tab);
        pred_plot->setObjectName(QString::fromUtf8("pred_plot"));
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(pred_plot->sizePolicy().hasHeightForWidth());
        pred_plot->setSizePolicy(sizePolicy);

        verticalLayout_2->addWidget(pred_plot);

        tabWidget->addTab(tab, QString());
        tab_2 = new QWidget();
        tab_2->setObjectName(QString::fromUtf8("tab_2"));
        verticalLayout_3 = new QVBoxLayout(tab_2);
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        formLayout_2 = new QFormLayout();
        formLayout_2->setObjectName(QString::fromUtf8("formLayout_2"));
        hp_x_axis = new QLabel(tab_2);
        hp_x_axis->setObjectName(QString::fromUtf8("hp_x_axis"));

        formLayout_2->setWidget(0, QFormLayout::LabelRole, hp_x_axis);

        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        label_4 = new QLabel(tab_2);
        label_4->setObjectName(QString::fromUtf8("label_4"));

        horizontalLayout_3->addWidget(label_4);

        hp_x_min = new QDoubleSpinBox(tab_2);
        hp_x_min->setObjectName(QString::fromUtf8("hp_x_min"));
        hp_x_min->setMinimum(-100.000000000000000);
        hp_x_min->setMaximum(100.000000000000000);
        hp_x_min->setValue(0.010000000000000);

        horizontalLayout_3->addWidget(hp_x_min);

        label_5 = new QLabel(tab_2);
        label_5->setObjectName(QString::fromUtf8("label_5"));

        horizontalLayout_3->addWidget(label_5);

        hp_x_max = new QDoubleSpinBox(tab_2);
        hp_x_max->setObjectName(QString::fromUtf8("hp_x_max"));
        hp_x_max->setMinimum(-100.000000000000000);
        hp_x_max->setMaximum(9999999.000000000000000);
        hp_x_max->setValue(100.000000000000000);

        horizontalLayout_3->addWidget(hp_x_max);

        hp_x_div_lbl = new QLabel(tab_2);
        hp_x_div_lbl->setObjectName(QString::fromUtf8("hp_x_div_lbl"));

        horizontalLayout_3->addWidget(hp_x_div_lbl);

        hp_x_div = new QSpinBox(tab_2);
        hp_x_div->setObjectName(QString::fromUtf8("hp_x_div"));
        hp_x_div->setMinimum(5);
        hp_x_div->setMaximum(256);
        hp_x_div->setValue(25);

        horizontalLayout_3->addWidget(hp_x_div);


        formLayout_2->setLayout(0, QFormLayout::FieldRole, horizontalLayout_3);

        label_7 = new QLabel(tab_2);
        label_7->setObjectName(QString::fromUtf8("label_7"));

        formLayout_2->setWidget(1, QFormLayout::LabelRole, label_7);

        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setObjectName(QString::fromUtf8("horizontalLayout_4"));
        hp_y_min_lbl = new QLabel(tab_2);
        hp_y_min_lbl->setObjectName(QString::fromUtf8("hp_y_min_lbl"));

        horizontalLayout_4->addWidget(hp_y_min_lbl);

        hp_y_min = new QDoubleSpinBox(tab_2);
        hp_y_min->setObjectName(QString::fromUtf8("hp_y_min"));
        hp_y_min->setMinimum(-100.000000000000000);
        hp_y_min->setMaximum(100.000000000000000);
        hp_y_min->setValue(0.010000000000000);

        horizontalLayout_4->addWidget(hp_y_min);

        hp_y_max_lbl = new QLabel(tab_2);
        hp_y_max_lbl->setObjectName(QString::fromUtf8("hp_y_max_lbl"));

        horizontalLayout_4->addWidget(hp_y_max_lbl);

        hp_y_max = new QDoubleSpinBox(tab_2);
        hp_y_max->setObjectName(QString::fromUtf8("hp_y_max"));
        hp_y_max->setMinimum(0.000000000000000);
        hp_y_max->setMaximum(999999.000000000000000);
        hp_y_max->setValue(10000.000000000000000);

        horizontalLayout_4->addWidget(hp_y_max);

        label_10 = new QLabel(tab_2);
        label_10->setObjectName(QString::fromUtf8("label_10"));

        horizontalLayout_4->addWidget(label_10);

        hp_y_div = new QSpinBox(tab_2);
        hp_y_div->setObjectName(QString::fromUtf8("hp_y_div"));
        hp_y_div->setMinimum(5);
        hp_y_div->setMaximum(256);
        hp_y_div->setValue(25);

        horizontalLayout_4->addWidget(hp_y_div);


        formLayout_2->setLayout(1, QFormLayout::FieldRole, horizontalLayout_4);


        verticalLayout_3->addLayout(formLayout_2);

        generate_lml = new QPushButton(tab_2);
        generate_lml->setObjectName(QString::fromUtf8("generate_lml"));

        verticalLayout_3->addWidget(generate_lml);

        lml_plot = new QCustomPlot(tab_2);
        lml_plot->setObjectName(QString::fromUtf8("lml_plot"));
        sizePolicy.setHeightForWidth(lml_plot->sizePolicy().hasHeightForWidth());
        lml_plot->setSizePolicy(sizePolicy);

        verticalLayout_3->addWidget(lml_plot);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        lml_min = new QDoubleSpinBox(tab_2);
        lml_min->setObjectName(QString::fromUtf8("lml_min"));

        horizontalLayout->addWidget(lml_min);

        lml_max = new QDoubleSpinBox(tab_2);
        lml_max->setObjectName(QString::fromUtf8("lml_max"));

        horizontalLayout->addWidget(lml_max);

        pos_lbl = new QLabel(tab_2);
        pos_lbl->setObjectName(QString::fromUtf8("pos_lbl"));

        horizontalLayout->addWidget(pos_lbl);


        verticalLayout_3->addLayout(horizontalLayout);

        tabWidget->addTab(tab_2, QString());

        verticalLayout->addWidget(tabWidget);


        retranslateUi(CSV2GPRWidet);

        tabWidget->setCurrentIndex(1);


        QMetaObject::connectSlotsByName(CSV2GPRWidet);
    } // setupUi

    void retranslateUi(QWidget *CSV2GPRWidet)
    {
        CSV2GPRWidet->setWindowTitle(QApplication::translate("CSV2GPRWidet", "Form", nullptr));
        open_csv_btn->setText(QApplication::translate("CSV2GPRWidet", "Open CSV", nullptr));
        label_6->setText(QApplication::translate("CSV2GPRWidet", "Block Size", nullptr));
        label->setText(QApplication::translate("CSV2GPRWidet", "x range", nullptr));
        label_2->setText(QApplication::translate("CSV2GPRWidet", "y value", nullptr));
        label_3->setText(QApplication::translate("CSV2GPRWidet", "divisions", nullptr));
        predict_btn->setText(QApplication::translate("CSV2GPRWidet", "Predict", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(tab), QApplication::translate("CSV2GPRWidet", "1D Plot", nullptr));
        hp_x_axis->setText(QApplication::translate("CSV2GPRWidet", "HP X Axis", nullptr));
        label_4->setText(QApplication::translate("CSV2GPRWidet", "min", nullptr));
        label_5->setText(QApplication::translate("CSV2GPRWidet", "max", nullptr));
        hp_x_div_lbl->setText(QApplication::translate("CSV2GPRWidet", "divisions", nullptr));
        label_7->setText(QApplication::translate("CSV2GPRWidet", "HP Y Axis", nullptr));
        hp_y_min_lbl->setText(QApplication::translate("CSV2GPRWidet", "min", nullptr));
        hp_y_max_lbl->setText(QApplication::translate("CSV2GPRWidet", "max", nullptr));
        label_10->setText(QApplication::translate("CSV2GPRWidet", "divisions", nullptr));
        generate_lml->setText(QApplication::translate("CSV2GPRWidet", "Generate", nullptr));
        pos_lbl->setText(QApplication::translate("CSV2GPRWidet", "TextLabel", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(tab_2), QApplication::translate("CSV2GPRWidet", "LML Plot", nullptr));
    } // retranslateUi

};

namespace Ui {
    class CSV2GPRWidet: public Ui_CSV2GPRWidet {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_CSV2GPR_WIDET_H
