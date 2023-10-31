/********************************************************************************
** Form generated from reading UI file 'bpslam_particle_widget.ui'
**
** Created by: Qt User Interface Compiler version 5.12.8
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_BPSLAM_PARTICLE_WIDGET_H
#define UI_BPSLAM_PARTICLE_WIDGET_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QFormLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_BpslamParticleWidget
{
public:
    QVBoxLayout *verticalLayout;
    QFormLayout *formLayout;
    QLabel *label_start_time;
    QLabel *label_end_time;
    QLabel *label_id;
    QLabel *id;
    QLabel *start_time;
    QLabel *end_time;
    QLabel *label_publish;
    QHBoxLayout *horizontalLayout;
    QPushButton *hypothesis_btn;
    QPushButton *recompute_gpr;
    QPushButton *gpr_btn;
    QPushButton *mean_btn;
    QPushButton *gpr_training_btn;
    QLabel *label;
    QLabel *boundary;
    QLabel *label_2;
    QLabel *likelihood;
    QLabel *label_3;
    QHBoxLayout *horizontalLayout_2;
    QPushButton *save_map_btn;

    void setupUi(QWidget *BpslamParticleWidget)
    {
        if (BpslamParticleWidget->objectName().isEmpty())
            BpslamParticleWidget->setObjectName(QString::fromUtf8("BpslamParticleWidget"));
        BpslamParticleWidget->resize(621, 195);
        verticalLayout = new QVBoxLayout(BpslamParticleWidget);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        formLayout = new QFormLayout();
        formLayout->setObjectName(QString::fromUtf8("formLayout"));
        label_start_time = new QLabel(BpslamParticleWidget);
        label_start_time->setObjectName(QString::fromUtf8("label_start_time"));
        QFont font;
        font.setBold(true);
        font.setWeight(75);
        label_start_time->setFont(font);

        formLayout->setWidget(1, QFormLayout::LabelRole, label_start_time);

        label_end_time = new QLabel(BpslamParticleWidget);
        label_end_time->setObjectName(QString::fromUtf8("label_end_time"));
        label_end_time->setFont(font);

        formLayout->setWidget(2, QFormLayout::LabelRole, label_end_time);

        label_id = new QLabel(BpslamParticleWidget);
        label_id->setObjectName(QString::fromUtf8("label_id"));
        label_id->setFont(font);

        formLayout->setWidget(0, QFormLayout::LabelRole, label_id);

        id = new QLabel(BpslamParticleWidget);
        id->setObjectName(QString::fromUtf8("id"));

        formLayout->setWidget(0, QFormLayout::FieldRole, id);

        start_time = new QLabel(BpslamParticleWidget);
        start_time->setObjectName(QString::fromUtf8("start_time"));

        formLayout->setWidget(1, QFormLayout::FieldRole, start_time);

        end_time = new QLabel(BpslamParticleWidget);
        end_time->setObjectName(QString::fromUtf8("end_time"));

        formLayout->setWidget(2, QFormLayout::FieldRole, end_time);

        label_publish = new QLabel(BpslamParticleWidget);
        label_publish->setObjectName(QString::fromUtf8("label_publish"));
        label_publish->setFont(font);

        formLayout->setWidget(5, QFormLayout::LabelRole, label_publish);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        hypothesis_btn = new QPushButton(BpslamParticleWidget);
        hypothesis_btn->setObjectName(QString::fromUtf8("hypothesis_btn"));

        horizontalLayout->addWidget(hypothesis_btn);

        recompute_gpr = new QPushButton(BpslamParticleWidget);
        recompute_gpr->setObjectName(QString::fromUtf8("recompute_gpr"));

        horizontalLayout->addWidget(recompute_gpr);

        gpr_btn = new QPushButton(BpslamParticleWidget);
        gpr_btn->setObjectName(QString::fromUtf8("gpr_btn"));

        horizontalLayout->addWidget(gpr_btn);

        mean_btn = new QPushButton(BpslamParticleWidget);
        mean_btn->setObjectName(QString::fromUtf8("mean_btn"));

        horizontalLayout->addWidget(mean_btn);

        gpr_training_btn = new QPushButton(BpslamParticleWidget);
        gpr_training_btn->setObjectName(QString::fromUtf8("gpr_training_btn"));

        horizontalLayout->addWidget(gpr_training_btn);


        formLayout->setLayout(5, QFormLayout::FieldRole, horizontalLayout);

        label = new QLabel(BpslamParticleWidget);
        label->setObjectName(QString::fromUtf8("label"));
        label->setFont(font);

        formLayout->setWidget(3, QFormLayout::LabelRole, label);

        boundary = new QLabel(BpslamParticleWidget);
        boundary->setObjectName(QString::fromUtf8("boundary"));

        formLayout->setWidget(3, QFormLayout::FieldRole, boundary);

        label_2 = new QLabel(BpslamParticleWidget);
        label_2->setObjectName(QString::fromUtf8("label_2"));
        label_2->setFont(font);

        formLayout->setWidget(4, QFormLayout::LabelRole, label_2);

        likelihood = new QLabel(BpslamParticleWidget);
        likelihood->setObjectName(QString::fromUtf8("likelihood"));

        formLayout->setWidget(4, QFormLayout::FieldRole, likelihood);

        label_3 = new QLabel(BpslamParticleWidget);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        formLayout->setWidget(6, QFormLayout::LabelRole, label_3);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        save_map_btn = new QPushButton(BpslamParticleWidget);
        save_map_btn->setObjectName(QString::fromUtf8("save_map_btn"));

        horizontalLayout_2->addWidget(save_map_btn);


        formLayout->setLayout(6, QFormLayout::FieldRole, horizontalLayout_2);


        verticalLayout->addLayout(formLayout);


        retranslateUi(BpslamParticleWidget);

        QMetaObject::connectSlotsByName(BpslamParticleWidget);
    } // setupUi

    void retranslateUi(QWidget *BpslamParticleWidget)
    {
        BpslamParticleWidget->setWindowTitle(QApplication::translate("BpslamParticleWidget", "Form", nullptr));
        label_start_time->setText(QApplication::translate("BpslamParticleWidget", "Start Time:", nullptr));
        label_end_time->setText(QApplication::translate("BpslamParticleWidget", "End Time:", nullptr));
        label_id->setText(QApplication::translate("BpslamParticleWidget", "ID:", nullptr));
        id->setText(QApplication::translate("BpslamParticleWidget", "0", nullptr));
        start_time->setText(QApplication::translate("BpslamParticleWidget", "0", nullptr));
        end_time->setText(QApplication::translate("BpslamParticleWidget", "0", nullptr));
        label_publish->setText(QApplication::translate("BpslamParticleWidget", "Publish:", nullptr));
        hypothesis_btn->setText(QApplication::translate("BpslamParticleWidget", "Hypothesis", nullptr));
        recompute_gpr->setText(QApplication::translate("BpslamParticleWidget", "Recompute GPR", nullptr));
        gpr_btn->setText(QApplication::translate("BpslamParticleWidget", "GPR Prediction", nullptr));
        mean_btn->setText(QApplication::translate("BpslamParticleWidget", "Mean Fn", nullptr));
        gpr_training_btn->setText(QApplication::translate("BpslamParticleWidget", "GPR Training", nullptr));
        label->setText(QApplication::translate("BpslamParticleWidget", "Boundary:", nullptr));
        boundary->setText(QApplication::translate("BpslamParticleWidget", "(0-0),(0,0)", nullptr));
        label_2->setText(QApplication::translate("BpslamParticleWidget", "Likelihood", nullptr));
        likelihood->setText(QApplication::translate("BpslamParticleWidget", "0", nullptr));
        label_3->setText(QApplication::translate("BpslamParticleWidget", "Save:", nullptr));
        save_map_btn->setText(QApplication::translate("BpslamParticleWidget", "Save Map..", nullptr));
    } // retranslateUi

};

namespace Ui {
    class BpslamParticleWidget: public Ui_BpslamParticleWidget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_BPSLAM_PARTICLE_WIDGET_H
