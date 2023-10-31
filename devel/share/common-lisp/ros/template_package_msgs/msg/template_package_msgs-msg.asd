
(cl:in-package :asdf)

(defsystem "template_package_msgs-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "example" :depends-on ("_package_example"))
    (:file "_package_example" :depends-on ("_package"))
  ))