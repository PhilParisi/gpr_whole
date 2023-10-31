/****************************************************************************
** Meta object code from reading C++ file 'bpslam_widget.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.12.8)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "include/sensor_stream_ros/ui/bpslam_widget.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'bpslam_widget.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.12.8. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_BPSlamWidget_t {
    QByteArrayData data[20];
    char stringdata0[373];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_BPSlamWidget_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_BPSlamWidget_t qt_meta_stringdata_BPSlamWidget = {
    {
QT_MOC_LITERAL(0, 0, 12), // "BPSlamWidget"
QT_MOC_LITERAL(1, 13, 16), // "uiUpdateComplete"
QT_MOC_LITERAL(2, 30, 0), // ""
QT_MOC_LITERAL(3, 31, 8), // "updateUI"
QT_MOC_LITERAL(4, 40, 12), // "updateValues"
QT_MOC_LITERAL(5, 53, 26), // "on_random_vars_itemChanged"
QT_MOC_LITERAL(6, 80, 16), // "QListWidgetItem*"
QT_MOC_LITERAL(7, 97, 4), // "item"
QT_MOC_LITERAL(8, 102, 36), // "on_leaf_particles_currentItem..."
QT_MOC_LITERAL(9, 139, 7), // "current"
QT_MOC_LITERAL(10, 147, 8), // "previous"
QT_MOC_LITERAL(11, 156, 28), // "on_pub_particle_tree_clicked"
QT_MOC_LITERAL(12, 185, 23), // "on_optimize_btn_clicked"
QT_MOC_LITERAL(13, 209, 28), // "on_generate_lml_plot_clicked"
QT_MOC_LITERAL(14, 238, 22), // "on_plotter_btn_clicked"
QT_MOC_LITERAL(15, 261, 23), // "on_min_lml_valueChanged"
QT_MOC_LITERAL(16, 285, 4), // "arg1"
QT_MOC_LITERAL(17, 290, 26), // "on_save_metric_btn_clicked"
QT_MOC_LITERAL(18, 317, 27), // "on_metrics_file_btn_clicked"
QT_MOC_LITERAL(19, 345, 27) // "on_metrics_file_textChanged"

    },
    "BPSlamWidget\0uiUpdateComplete\0\0updateUI\0"
    "updateValues\0on_random_vars_itemChanged\0"
    "QListWidgetItem*\0item\0"
    "on_leaf_particles_currentItemChanged\0"
    "current\0previous\0on_pub_particle_tree_clicked\0"
    "on_optimize_btn_clicked\0"
    "on_generate_lml_plot_clicked\0"
    "on_plotter_btn_clicked\0on_min_lml_valueChanged\0"
    "arg1\0on_save_metric_btn_clicked\0"
    "on_metrics_file_btn_clicked\0"
    "on_metrics_file_textChanged"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_BPSlamWidget[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
      13,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    0,   79,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       3,    0,   80,    2, 0x0a /* Public */,
       4,    0,   81,    2, 0x08 /* Private */,
       5,    1,   82,    2, 0x08 /* Private */,
       8,    2,   85,    2, 0x08 /* Private */,
      11,    0,   90,    2, 0x08 /* Private */,
      12,    0,   91,    2, 0x08 /* Private */,
      13,    0,   92,    2, 0x08 /* Private */,
      14,    0,   93,    2, 0x08 /* Private */,
      15,    1,   94,    2, 0x08 /* Private */,
      17,    0,   97,    2, 0x08 /* Private */,
      18,    0,   98,    2, 0x08 /* Private */,
      19,    1,   99,    2, 0x08 /* Private */,

 // signals: parameters
    QMetaType::Void,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, 0x80000000 | 6,    7,
    QMetaType::Void, 0x80000000 | 6, 0x80000000 | 6,    9,   10,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Double,   16,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::QString,   16,

       0        // eod
};

void BPSlamWidget::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<BPSlamWidget *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->uiUpdateComplete(); break;
        case 1: _t->updateUI(); break;
        case 2: _t->updateValues(); break;
        case 3: _t->on_random_vars_itemChanged((*reinterpret_cast< QListWidgetItem*(*)>(_a[1]))); break;
        case 4: _t->on_leaf_particles_currentItemChanged((*reinterpret_cast< QListWidgetItem*(*)>(_a[1])),(*reinterpret_cast< QListWidgetItem*(*)>(_a[2]))); break;
        case 5: _t->on_pub_particle_tree_clicked(); break;
        case 6: _t->on_optimize_btn_clicked(); break;
        case 7: _t->on_generate_lml_plot_clicked(); break;
        case 8: _t->on_plotter_btn_clicked(); break;
        case 9: _t->on_min_lml_valueChanged((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 10: _t->on_save_metric_btn_clicked(); break;
        case 11: _t->on_metrics_file_btn_clicked(); break;
        case 12: _t->on_metrics_file_textChanged((*reinterpret_cast< const QString(*)>(_a[1]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (BPSlamWidget::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&BPSlamWidget::uiUpdateComplete)) {
                *result = 0;
                return;
            }
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject BPSlamWidget::staticMetaObject = { {
    &QWidget::staticMetaObject,
    qt_meta_stringdata_BPSlamWidget.data,
    qt_meta_data_BPSlamWidget,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *BPSlamWidget::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *BPSlamWidget::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_BPSlamWidget.stringdata0))
        return static_cast<void*>(this);
    return QWidget::qt_metacast(_clname);
}

int BPSlamWidget::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 13)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 13;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 13)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 13;
    }
    return _id;
}

// SIGNAL 0
void BPSlamWidget::uiUpdateComplete()
{
    QMetaObject::activate(this, &staticMetaObject, 0, nullptr);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
