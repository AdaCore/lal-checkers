with Ada.Text_IO; use Ada.Text_IO;
procedure Test is
   X : aliased Integer := 12;

   procedure F is
      Y : access Integer := X'Access;
   begin
      Y.all := 2;
   end F;
begin
   F;
   F;
end Ex1;
