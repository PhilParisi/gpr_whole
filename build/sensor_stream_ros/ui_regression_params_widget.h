/********************************************************************************
** Form generated from reading UI file 'regression_params_widget.ui'
**
** Created by: Qt User Interface Compiler version 5.12.8
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_REGRESSION_PARAMS_WIDGET_H
#define UI_REGRESSION_PARAMS_WIDGET_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QFormLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_RegressionParamsWidget
{
public:
    QVBoxLayout *verticalLayout;
    QLabel *label;
    QFormLayout *params;
    QLabel *label_2;
    QDoubleSpinBox *sensor_var;
    QLabel *label_3;
    QDoubleSpinBox *nnz_thresh;
    QLabel *kernel_type;
    QVBoxLayout *hyperparams;

    void setupUi(QWidget *RegressionParamsWidget)
    {
        if (RegressionParamsWidget->objectName().isEmpty())
            RegressionParamsWidget->setObjectName(QString::fromUtf8("RegressionParamsWidget"));
        RegressionParamsWidget->resize(400, 221);
        verticalLayout = new QVBoxLayout(RegressionParamsWidget);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        label = new QLabel(RegressionParamsWidget);
        label->setObjectName(QString::fromUtf8("label"));
        QFont font;
        font.setPointSize(12);
        font.setBold(true);
        font.setWeight(75);
        label->setFont(font);

        verticalLayout->addWidget(label);

        params = new QFormLayout();
        params->setObjectName(QString::fromUtf8("params"));
        label_2 = new QLabel(RegressionParamsWidget);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        params->setWidget(0, QFormLayout::LabelRole, label_2);

        sensor_var = new QDoubleSpinBox(RegressionParamsWidget);
        sensor_var->setObjectName(QString::fromUtf8("sensor_var"));
        sensor_var->setDecimals(5);
        sensor_var->setSingleStep(0.100000000000000);

        params->setWidget(0, QFormLayout::FieldRole, sensor_var);

        label_3 = new QLabel(RegressionParamsWidget);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        params->setWidget(1, QFormLayout::LabelRole, label_3);

        nnz_thresh = new QDoubleSpinBox(RegressionParamsWidget);
        nnz_thresh->setObjectName(QString::fromUtf8("nnz_thresh"));
        nnz_thresh->setDecimals(8);
        nnz_thresh->setMaximum(10.000000000000000);
        nnz_thresh->setSingleStep(0.000100000000000);

        params->setWidget(1, QFormLayout::FieldRole, nnz_thresh);


        verticalLayout->addLayout(params);

        kernel_type = new QLabel(RegressionParamsWidget);
        kernel_type->setObjectName(QString::fromUtf8("kernel_type"));
        kernel_type->setFont(font);

        verticalLayout->addWidget(kernel_type);

        hyperparams = new QVBoxLayout();
        hyperparams->setObjectName(QString::fromUtf8("hyperparams"));

        verticalLayout->addLayout(hyperparams);


        retranslateUi(RegressionParamsWidget);

        QMetaObject::connectSlotsByName(RegressionParamsWidget);
    } // setupUi

    void retranslateUi(QWidget *RegressionParamsWidget)
    {
        RegressionParamsWidget->setWindowTitle(QApplication::translate("RegressionParamsWidget", "Form", nullptr));
        label->setText(QApplication::translate("RegressionParamsWidget", "Parameters", nullptr));
        label_2->setText(QApplication::translate("RegressionParamsWidget", "Sensor Variance", nullptr));
        label_3->setText(QApplication::translate("RegressionParamsWidget", "NNZ Threshold", nullptr));
        kernel_type->setText(QApplication::translate("RegressionParamsWidget", "Hyperparemeters", nullptr));
    } // retranslateUi

};

namespace Ui {
    class RegressionParamsWidget: public Ui_RegressionParamsWidget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_REGRESSION_PARAMS_WIDGET_H
