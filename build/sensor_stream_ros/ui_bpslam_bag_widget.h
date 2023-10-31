/********************************************************************************
** Form generated from reading UI file 'bpslam_bag_widget.ui'
**
** Created by: Qt User Interface Compiler version 5.12.8
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_BPSLAM_BAG_WIDGET_H
#define UI_BPSLAM_BAG_WIDGET_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QFormLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>
#include "include/sensor_stream_ros/ui/bpslam_widget.h"

QT_BEGIN_NAMESPACE

class Ui_BPSlamBagWidget
{
public:
    QVBoxLayout *verticalLayout;
    QFormLayout *file_selectors;
    QPushButton *bag_button;
    QLineEdit *bag_file;
    QPushButton *urdf_button;
    QLineEdit *urdf_file;
    QLabel *label;
    BPSlamWidget *bpslam_widget;
    QPushButton *start_btn;
    QHBoxLayout *control_btns;
    QPushButton *run_btn;
    QPushButton *pause_btn;
    QPushButton *step_btn;
    QPushButton *gpr_btn;
    QPushButton *cull_btn;

    void setupUi(QWidget *BPSlamBagWidget)
    {
        if (BPSlamBagWidget->objectName().isEmpty())
            BPSlamBagWidget->setObjectName(QString::fromUtf8("BPSlamBagWidget"));
        BPSlamBagWidget->resize(487, 1114);
        verticalLayout = new QVBoxLayout(BPSlamBagWidget);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        file_selectors = new QFormLayout();
        file_selectors->setObjectName(QString::fromUtf8("file_selectors"));
        bag_button = new QPushButton(BPSlamBagWidget);
        bag_button->setObjectName(QString::fromUtf8("bag_button"));

        file_selectors->setWidget(0, QFormLayout::LabelRole, bag_button);

        bag_file = new QLineEdit(BPSlamBagWidget);
        bag_file->setObjectName(QString::fromUtf8("bag_file"));

        file_selectors->setWidget(0, QFormLayout::FieldRole, bag_file);

        urdf_button = new QPushButton(BPSlamBagWidget);
        urdf_button->setObjectName(QString::fromUtf8("urdf_button"));

        file_selectors->setWidget(1, QFormLayout::LabelRole, urdf_button);

        urdf_file = new QLineEdit(BPSlamBagWidget);
        urdf_file->setObjectName(QString::fromUtf8("urdf_file"));

        file_selectors->setWidget(1, QFormLayout::FieldRole, urdf_file);


        verticalLayout->addLayout(file_selectors);

        label = new QLabel(BPSlamBagWidget);
        label->setObjectName(QString::fromUtf8("label"));

        verticalLayout->addWidget(label);

        bpslam_widget = new BPSlamWidget(BPSlamBagWidget);
        bpslam_widget->setObjectName(QString::fromUtf8("bpslam_widget"));
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(bpslam_widget->sizePolicy().hasHeightForWidth());
        bpslam_widget->setSizePolicy(sizePolicy);

        verticalLayout->addWidget(bpslam_widget);

        start_btn = new QPushButton(BPSlamBagWidget);
        start_btn->setObjectName(QString::fromUtf8("start_btn"));

        verticalLayout->addWidget(start_btn);

        control_btns = new QHBoxLayout();
        control_btns->setObjectName(QString::fromUtf8("control_btns"));
        run_btn = new QPushButton(BPSlamBagWidget);
        run_btn->setObjectName(QString::fromUtf8("run_btn"));

        control_btns->addWidget(run_btn);

        pause_btn = new QPushButton(BPSlamBagWidget);
        pause_btn->setObjectName(QString::fromUtf8("pause_btn"));

        control_btns->addWidget(pause_btn);

        step_btn = new QPushButton(BPSlamBagWidget);
        step_btn->setObjectName(QString::fromUtf8("step_btn"));

        control_btns->addWidget(step_btn);

        gpr_btn = new QPushButton(BPSlamBagWidget);
        gpr_btn->setObjectName(QString::fromUtf8("gpr_btn"));

        control_btns->addWidget(gpr_btn);

        cull_btn = new QPushButton(BPSlamBagWidget);
        cull_btn->setObjectName(QString::fromUtf8("cull_btn"));

        control_btns->addWidget(cull_btn);


        verticalLayout->addLayout(control_btns);


        retranslateUi(BPSlamBagWidget);

        QMetaObject::connectSlotsByName(BPSlamBagWidget);
    } // setupUi

    void retranslateUi(QWidget *BPSlamBagWidget)
    {
        BPSlamBagWidget->setWindowTitle(QApplication::translate("BPSlamBagWidget", "Form", nullptr));
        bag_button->setText(QApplication::translate("BPSlamBagWidget", "Bag File...", nullptr));
        urdf_button->setText(QApplication::translate("BPSlamBagWidget", "URDF File...", nullptr));
        label->setText(QApplication::translate("BPSlamBagWidget", "BPSlam Config", nullptr));
        start_btn->setText(QApplication::translate("BPSlamBagWidget", "Start", nullptr));
        run_btn->setText(QApplication::translate("BPSlamBagWidget", "Run", nullptr));
        pause_btn->setText(QApplication::translate("BPSlamBagWidget", "Pause", nullptr));
        step_btn->setText(QApplication::translate("BPSlamBagWidget", "Step Once", nullptr));
        gpr_btn->setText(QApplication::translate("BPSlamBagWidget", "Compute GPR", nullptr));
        cull_btn->setText(QApplication::translate("BPSlamBagWidget", "Cull Particles", nullptr));
    } // retranslateUi

};

namespace Ui {
    class BPSlamBagWidget: public Ui_BPSlamBagWidget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_BPSLAM_BAG_WIDGET_H
