/********************************************************************************
** Form generated from reading UI file 'bpslam_widget.ui'
**
** Created by: Qt User Interface Compiler version 5.12.8
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_BPSLAM_WIDGET_H
#define UI_BPSLAM_WIDGET_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QDockWidget>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QFormLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QListWidget>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QToolButton>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>
#include <include/sensor_stream_ros/ui/regression_params_widget.h>
#include "include/sensor_stream_ros/third_party/qcustomplot.h"
#include "include/sensor_stream_ros/ui/bpslam_particle_widget.h"

QT_BEGIN_NAMESPACE

class Ui_BPSlamWidget
{
public:
    QVBoxLayout *verticalLayout;
    QHBoxLayout *horizontalLayout_2;
    QLabel *label_10;
    QLineEdit *metrics_file;
    QToolButton *metrics_file_btn;
    QFormLayout *formLayout;
    QLabel *label;
    QDoubleSpinBox *particle_lifespan;
    QLabel *label_2;
    QSpinBox *particle_n_children;
    QLabel *label_3;
    QSpinBox *max_particles;
    QLabel *label_6;
    QSpinBox *steps_between_cull;
    QLabel *label_4;
    QListWidget *random_vars;
    QLabel *label_7;
    QLabel *num_leaf_particles;
    QLabel *label_5;
    QListWidget *leaf_particles;
    QLabel *label_8;
    QDoubleSpinBox *min_model_particle_age;
    QLabel *ekf_uncertainty_multiplier_label;
    QDoubleSpinBox *ekf_uncertainty_multiplier;
    QLabel *label_9;
    QSpinBox *min_particles;
    QHBoxLayout *horizontalLayout;
    QPushButton *plotter_btn;
    QPushButton *save_metric_btn;
    QPushButton *pub_particle_tree;
    QTabWidget *tabWidget;
    QWidget *particle_tab;
    QVBoxLayout *verticalLayout_2;
    BpslamParticleWidget *particle_details;
    QWidget *regression_tab;
    QVBoxLayout *verticalLayout_3;
    RegressionParamsWidget *regression_params;
    QPushButton *optimize_btn;
    QTabWidget *tabWidget_2;
    QWidget *error_plot_tab;
    QVBoxLayout *verticalLayout_5;
    QDockWidget *dockWidget;
    QCustomPlot *error_plot;
    QWidget *lml_tab;
    QVBoxLayout *verticalLayout_4;
    QCustomPlot *lml_plot;
    QDoubleSpinBox *min_lml;
    QPushButton *generate_lml_plot;

    void setupUi(QWidget *BPSlamWidget)
    {
        if (BPSlamWidget->objectName().isEmpty())
            BPSlamWidget->setObjectName(QString::fromUtf8("BPSlamWidget"));
        BPSlamWidget->resize(382, 1147);
        verticalLayout = new QVBoxLayout(BPSlamWidget);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        label_10 = new QLabel(BPSlamWidget);
        label_10->setObjectName(QString::fromUtf8("label_10"));

        horizontalLayout_2->addWidget(label_10);

        metrics_file = new QLineEdit(BPSlamWidget);
        metrics_file->setObjectName(QString::fromUtf8("metrics_file"));

        horizontalLayout_2->addWidget(metrics_file);

        metrics_file_btn = new QToolButton(BPSlamWidget);
        metrics_file_btn->setObjectName(QString::fromUtf8("metrics_file_btn"));

        horizontalLayout_2->addWidget(metrics_file_btn);


        verticalLayout->addLayout(horizontalLayout_2);

        formLayout = new QFormLayout();
        formLayout->setObjectName(QString::fromUtf8("formLayout"));
        label = new QLabel(BPSlamWidget);
        label->setObjectName(QString::fromUtf8("label"));

        formLayout->setWidget(0, QFormLayout::LabelRole, label);

        particle_lifespan = new QDoubleSpinBox(BPSlamWidget);
        particle_lifespan->setObjectName(QString::fromUtf8("particle_lifespan"));
        particle_lifespan->setMinimum(0.100000000000000);
        particle_lifespan->setMaximum(600.000000000000000);
        particle_lifespan->setValue(4.000000000000000);

        formLayout->setWidget(0, QFormLayout::FieldRole, particle_lifespan);

        label_2 = new QLabel(BPSlamWidget);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        formLayout->setWidget(1, QFormLayout::LabelRole, label_2);

        particle_n_children = new QSpinBox(BPSlamWidget);
        particle_n_children->setObjectName(QString::fromUtf8("particle_n_children"));
        particle_n_children->setMinimum(1);
        particle_n_children->setMaximum(256);
        particle_n_children->setValue(2);

        formLayout->setWidget(1, QFormLayout::FieldRole, particle_n_children);

        label_3 = new QLabel(BPSlamWidget);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        formLayout->setWidget(2, QFormLayout::LabelRole, label_3);

        max_particles = new QSpinBox(BPSlamWidget);
        max_particles->setObjectName(QString::fromUtf8("max_particles"));
        max_particles->setMinimum(1);
        max_particles->setMaximum(10000);
        max_particles->setValue(16);

        formLayout->setWidget(2, QFormLayout::FieldRole, max_particles);

        label_6 = new QLabel(BPSlamWidget);
        label_6->setObjectName(QString::fromUtf8("label_6"));

        formLayout->setWidget(4, QFormLayout::LabelRole, label_6);

        steps_between_cull = new QSpinBox(BPSlamWidget);
        steps_between_cull->setObjectName(QString::fromUtf8("steps_between_cull"));
        steps_between_cull->setMaximum(999999999);
        steps_between_cull->setValue(6);

        formLayout->setWidget(4, QFormLayout::FieldRole, steps_between_cull);

        label_4 = new QLabel(BPSlamWidget);
        label_4->setObjectName(QString::fromUtf8("label_4"));

        formLayout->setWidget(7, QFormLayout::LabelRole, label_4);

        random_vars = new QListWidget(BPSlamWidget);
        random_vars->setObjectName(QString::fromUtf8("random_vars"));
        QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(random_vars->sizePolicy().hasHeightForWidth());
        random_vars->setSizePolicy(sizePolicy);
        random_vars->setMaximumSize(QSize(16777215, 100));

        formLayout->setWidget(7, QFormLayout::FieldRole, random_vars);

        label_7 = new QLabel(BPSlamWidget);
        label_7->setObjectName(QString::fromUtf8("label_7"));

        formLayout->setWidget(8, QFormLayout::LabelRole, label_7);

        num_leaf_particles = new QLabel(BPSlamWidget);
        num_leaf_particles->setObjectName(QString::fromUtf8("num_leaf_particles"));

        formLayout->setWidget(8, QFormLayout::FieldRole, num_leaf_particles);

        label_5 = new QLabel(BPSlamWidget);
        label_5->setObjectName(QString::fromUtf8("label_5"));

        formLayout->setWidget(9, QFormLayout::LabelRole, label_5);

        leaf_particles = new QListWidget(BPSlamWidget);
        leaf_particles->setObjectName(QString::fromUtf8("leaf_particles"));

        formLayout->setWidget(9, QFormLayout::FieldRole, leaf_particles);

        label_8 = new QLabel(BPSlamWidget);
        label_8->setObjectName(QString::fromUtf8("label_8"));

        formLayout->setWidget(5, QFormLayout::LabelRole, label_8);

        min_model_particle_age = new QDoubleSpinBox(BPSlamWidget);
        min_model_particle_age->setObjectName(QString::fromUtf8("min_model_particle_age"));
        min_model_particle_age->setMaximum(600.000000000000000);
        min_model_particle_age->setValue(180.000000000000000);

        formLayout->setWidget(5, QFormLayout::FieldRole, min_model_particle_age);

        ekf_uncertainty_multiplier_label = new QLabel(BPSlamWidget);
        ekf_uncertainty_multiplier_label->setObjectName(QString::fromUtf8("ekf_uncertainty_multiplier_label"));

        formLayout->setWidget(6, QFormLayout::LabelRole, ekf_uncertainty_multiplier_label);

        ekf_uncertainty_multiplier = new QDoubleSpinBox(BPSlamWidget);
        ekf_uncertainty_multiplier->setObjectName(QString::fromUtf8("ekf_uncertainty_multiplier"));
        ekf_uncertainty_multiplier->setDecimals(4);
        ekf_uncertainty_multiplier->setValue(1.000000000000000);

        formLayout->setWidget(6, QFormLayout::FieldRole, ekf_uncertainty_multiplier);

        label_9 = new QLabel(BPSlamWidget);
        label_9->setObjectName(QString::fromUtf8("label_9"));

        formLayout->setWidget(3, QFormLayout::LabelRole, label_9);

        min_particles = new QSpinBox(BPSlamWidget);
        min_particles->setObjectName(QString::fromUtf8("min_particles"));

        formLayout->setWidget(3, QFormLayout::FieldRole, min_particles);


        verticalLayout->addLayout(formLayout);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        plotter_btn = new QPushButton(BPSlamWidget);
        plotter_btn->setObjectName(QString::fromUtf8("plotter_btn"));

        horizontalLayout->addWidget(plotter_btn);

        save_metric_btn = new QPushButton(BPSlamWidget);
        save_metric_btn->setObjectName(QString::fromUtf8("save_metric_btn"));

        horizontalLayout->addWidget(save_metric_btn);


        verticalLayout->addLayout(horizontalLayout);

        pub_particle_tree = new QPushButton(BPSlamWidget);
        pub_particle_tree->setObjectName(QString::fromUtf8("pub_particle_tree"));

        verticalLayout->addWidget(pub_particle_tree);

        tabWidget = new QTabWidget(BPSlamWidget);
        tabWidget->setObjectName(QString::fromUtf8("tabWidget"));
        tabWidget->setTabPosition(QTabWidget::North);
        tabWidget->setTabShape(QTabWidget::Rounded);
        particle_tab = new QWidget();
        particle_tab->setObjectName(QString::fromUtf8("particle_tab"));
        verticalLayout_2 = new QVBoxLayout(particle_tab);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        particle_details = new BpslamParticleWidget(particle_tab);
        particle_details->setObjectName(QString::fromUtf8("particle_details"));

        verticalLayout_2->addWidget(particle_details);

        tabWidget->addTab(particle_tab, QString());
        regression_tab = new QWidget();
        regression_tab->setObjectName(QString::fromUtf8("regression_tab"));
        verticalLayout_3 = new QVBoxLayout(regression_tab);
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        regression_params = new RegressionParamsWidget(regression_tab);
        regression_params->setObjectName(QString::fromUtf8("regression_params"));

        verticalLayout_3->addWidget(regression_params);

        tabWidget->addTab(regression_tab, QString());

        verticalLayout->addWidget(tabWidget);

        optimize_btn = new QPushButton(BPSlamWidget);
        optimize_btn->setObjectName(QString::fromUtf8("optimize_btn"));

        verticalLayout->addWidget(optimize_btn);

        tabWidget_2 = new QTabWidget(BPSlamWidget);
        tabWidget_2->setObjectName(QString::fromUtf8("tabWidget_2"));
        error_plot_tab = new QWidget();
        error_plot_tab->setObjectName(QString::fromUtf8("error_plot_tab"));
        verticalLayout_5 = new QVBoxLayout(error_plot_tab);
        verticalLayout_5->setObjectName(QString::fromUtf8("verticalLayout_5"));
        dockWidget = new QDockWidget(error_plot_tab);
        dockWidget->setObjectName(QString::fromUtf8("dockWidget"));
        error_plot = new QCustomPlot();
        error_plot->setObjectName(QString::fromUtf8("error_plot"));
        dockWidget->setWidget(error_plot);

        verticalLayout_5->addWidget(dockWidget);

        tabWidget_2->addTab(error_plot_tab, QString());
        lml_tab = new QWidget();
        lml_tab->setObjectName(QString::fromUtf8("lml_tab"));
        verticalLayout_4 = new QVBoxLayout(lml_tab);
        verticalLayout_4->setObjectName(QString::fromUtf8("verticalLayout_4"));
        lml_plot = new QCustomPlot(lml_tab);
        lml_plot->setObjectName(QString::fromUtf8("lml_plot"));

        verticalLayout_4->addWidget(lml_plot);

        min_lml = new QDoubleSpinBox(lml_tab);
        min_lml->setObjectName(QString::fromUtf8("min_lml"));
        min_lml->setMinimum(-9999999999.000000000000000);
        min_lml->setMaximum(99999999999.000000000000000);
        min_lml->setSingleStep(100.000000000000000);

        verticalLayout_4->addWidget(min_lml);

        generate_lml_plot = new QPushButton(lml_tab);
        generate_lml_plot->setObjectName(QString::fromUtf8("generate_lml_plot"));

        verticalLayout_4->addWidget(generate_lml_plot);

        tabWidget_2->addTab(lml_tab, QString());

        verticalLayout->addWidget(tabWidget_2);


        retranslateUi(BPSlamWidget);

        tabWidget->setCurrentIndex(0);
        tabWidget_2->setCurrentIndex(1);


        QMetaObject::connectSlotsByName(BPSlamWidget);
    } // setupUi

    void retranslateUi(QWidget *BPSlamWidget)
    {
        BPSlamWidget->setWindowTitle(QApplication::translate("BPSlamWidget", "Form", nullptr));
        label_10->setText(QApplication::translate("BPSlamWidget", "Metrics File", nullptr));
        metrics_file_btn->setText(QApplication::translate("BPSlamWidget", "...", nullptr));
        label->setText(QApplication::translate("BPSlamWidget", "particle.lifespan (s)", nullptr));
        label_2->setText(QApplication::translate("BPSlamWidget", "particle.n_children", nullptr));
        label_3->setText(QApplication::translate("BPSlamWidget", "max_particles", nullptr));
        label_6->setText(QApplication::translate("BPSlamWidget", "steps_between_cull", nullptr));
        label_4->setText(QApplication::translate("BPSlamWidget", "random_vars", nullptr));
        label_7->setText(QApplication::translate("BPSlamWidget", "num Leaf Particles", nullptr));
        num_leaf_particles->setText(QApplication::translate("BPSlamWidget", "0", nullptr));
        label_5->setText(QApplication::translate("BPSlamWidget", "leaf particles", nullptr));
        label_8->setText(QApplication::translate("BPSlamWidget", "min_model_particle_age (s)", nullptr));
        ekf_uncertainty_multiplier_label->setText(QApplication::translate("BPSlamWidget", "ekf_uncertainty_multiplier", nullptr));
        label_9->setText(QApplication::translate("BPSlamWidget", "min_particles", nullptr));
        plotter_btn->setText(QApplication::translate("BPSlamWidget", "Plotter...", nullptr));
        save_metric_btn->setText(QApplication::translate("BPSlamWidget", "Save Metrics...", nullptr));
        pub_particle_tree->setText(QApplication::translate("BPSlamWidget", "Publish Particle Tree", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(particle_tab), QApplication::translate("BPSlamWidget", "Particle Info", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(regression_tab), QApplication::translate("BPSlamWidget", "Regression Params", nullptr));
        optimize_btn->setText(QApplication::translate("BPSlamWidget", "\342\232\241  Optimize HP \342\232\241 ", nullptr));
        tabWidget_2->setTabText(tabWidget_2->indexOf(error_plot_tab), QApplication::translate("BPSlamWidget", "Tab 1", nullptr));
        generate_lml_plot->setText(QApplication::translate("BPSlamWidget", "Regenerate LML Plot", nullptr));
        tabWidget_2->setTabText(tabWidget_2->indexOf(lml_tab), QApplication::translate("BPSlamWidget", "Tab 2", nullptr));
    } // retranslateUi

};

namespace Ui {
    class BPSlamWidget: public Ui_BPSlamWidget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_BPSLAM_WIDGET_H
