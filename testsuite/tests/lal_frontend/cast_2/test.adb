with Ada.Text_IO; use Ada.Text_IO;

procedure Test is
    procedure Foo (I : out Integer) is
    begin
       I := -2;
    end Foo;

    X : Natural := 0;
begin
    Foo (Integer (X));
    Put_Line (X'Image);
end Test;