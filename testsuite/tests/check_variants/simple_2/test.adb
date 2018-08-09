procedure Test is
   type My_Variant(x : Integer) is record
      case x is
         when 0 .. 500 =>
            case x is
               when 0 .. 250 =>
                  z : Integer;
               when others =>
                  w : Integer;
            end case;
         when others =>
            y : Boolean;
      end case;
   end record;
   x : My_Variant := (x => 120);
   res : Integer;
begin
   res := x.w;
end Ex1;
